import Foundation

#if canImport(CoreAI)
import CoreAI

/// In-process generator on `CoreAI.framework` for `SmolLM2Stateful.aimodel`
/// (`llm_ai.convert`): `main_prefill_t<N>` fills the KV caches from a prompt
/// in N-token chunks (the last one left-padded), then `main_decode` runs one
/// token per call. The caches are NDArrays owned by the generator and passed
/// as *states*, exactly like `SnakeCoreAI.ModelPlayer`; greedy sampling is
/// an argmax over the vocabulary logits of the newest token.
///
/// Prompts are token ids. `generate(prompt:)` looks the text up in
/// `prompt_ids.json` until a Swift BPE encoder exists.
@available(macOS 27, iOS 27, *)
public final class ModelGenerator: TokenGenerator {
    public let label: String
    public let loadMS: Double
    public let decoder: ByteLevelDecoder
    public let eosID: Int
    public let maxContext: Int
    /// "fp16" or "fp32", from the KV-cache state's scalar type.
    public let precision: String
    private let prompts: PromptIDs?
    private let engine: Engine

    /// Default specialization puts the fp16 graph on the Neural Engine,
    /// which fails to load it on an 8 GB M2 (docs/coreai-ecosystem.md,
    /// gotcha 17); `.gpu` is the working default here.
    public static let defaultOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)

    public init(modelURL: URL, tokenizerURL: URL, prompts: PromptIDs? = nil, eosID: Int = 2,
                options: SpecializationOptions = ModelGenerator.defaultOptions) async throws {
        let start = ContinuousClock.now
        self.engine = try await Engine(modelURL: modelURL, options: options, padID: prompts?.padId ?? 0)
        let elapsed = ContinuousClock.now - start
        self.loadMS = Double(elapsed.components.seconds) * 1e3 + Double(elapsed.components.attoseconds) / 1e15
        self.decoder = try ByteLevelDecoder(tokenizerJSON: tokenizerURL)
        self.prompts = prompts
        self.eosID = prompts?.eosId ?? eosID
        self.maxContext = engine.maxContext
        self.precision = engine.precision
        let prefill = engine.prefillLength
        self.label = "CoreAI.framework (in-process) — \(modelURL.lastPathComponent), prefill t\(prefill) + decode"
    }

    public func reset() async throws { await engine.reset() }

    /// Tokens already in the cache.
    public var position: Int { get async { await engine.position } }

    public func generate(prompt: String, maxTokens: Int) -> AsyncThrowingStream<GeneratedToken, Error> {
        guard let ids = prompts?.ids(for: prompt) else {
            return AsyncThrowingStream { $0.finish(throwing: GeneratorError.untokenizedPrompt(prompt)) }
        }
        return generate(promptIDs: ids, maxTokens: maxTokens)
    }

    /// Append `promptIDs` to the conversation and stream up to `maxTokens`
    /// greedy tokens. `stopAtEOS: false` always produces `maxTokens` (bench).
    /// Each token's `text` is the reply decoded so far, minus what earlier
    /// tokens already produced, so multi-byte characters are never split.
    public func generate(promptIDs: [Int], maxTokens: Int, stopAtEOS: Bool = true) -> AsyncThrowingStream<GeneratedToken, Error> {
        let engine = self.engine, decoder = self.decoder, eosID = self.eosID
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var clock = ContinuousClock.now
                    var next = try await engine.prefill(promptIDs)
                    var reply: [Int] = [], emitted = ""
                    for _ in 0..<maxTokens {
                        let ms = Self.ms(since: clock)
                        if stopAtEOS && next == eosID { break }
                        reply.append(next)
                        let text = decoder.decode(reply)
                        var piece = ""
                        if !text.hasSuffix("\u{FFFD}") {
                            piece = String(text.dropFirst(emitted.count))
                            emitted = text
                        }
                        continuation.yield(GeneratedToken(id: next, text: piece, ms: ms))
                        if reply.count == maxTokens { break }
                        try Task.checkCancellation()
                        clock = ContinuousClock.now
                        next = try await engine.decode(next)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func ms(since start: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - start
        return Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15
    }
}

/// Owns the functions and the caches; an actor so the caches are only ever
/// touched by one call at a time.
@available(macOS 27, iOS 27, *)
actor Engine {
    let prefillFunction: InferenceFunction
    let decodeFunction: InferenceFunction
    let prefillLength: Int
    let maxContext: Int
    let padID: Int
    let precision: String
    private let cacheShape: [Int]
    private let cacheType: NDArray.ScalarType
    // Optional so a call can move them out (see `run`); nil only during a call.
    private var keyCache: NDArray?
    private var valueCache: NDArray?
    private(set) var position = 0

    init(modelURL: URL, options: SpecializationOptions, padID: Int) async throws {
        let model = try await AIModel(contentsOf: modelURL, options: options)
        // Static prefill functions are named main_prefill_t<N> (gotcha 8); take the largest.
        let prefills = model.functionNames.compactMap { name -> (String, Int)? in
            guard name.hasPrefix("main_prefill_t"), let n = Int(name.dropFirst("main_prefill_t".count)) else { return nil }
            return (name, n)
        }
        guard let (prefillName, length) = prefills.max(by: { $0.1 < $1.1 }) else {
            throw GeneratorError.missingFunction("main_prefill_t*")
        }
        guard let prefill = try model.loadFunction(named: prefillName) else { throw GeneratorError.missingFunction(prefillName) }
        guard let decode = try model.loadFunction(named: "main_decode") else { throw GeneratorError.missingFunction("main_decode") }
        self.prefillFunction = prefill
        self.decodeFunction = decode
        self.prefillLength = length
        self.padID = padID

        // Shape and scalar type come from the state descriptor rather than
        // being hard-coded: [layers, 1, kv_heads, max_seq_len, head_dim].
        guard case .ndArray(let d)? = decode.descriptor.stateDescriptor(of: "keyCache") else {
            throw GeneratorError.missingState("keyCache")
        }
        let probe = NDArray(descriptor: d)
        self.cacheShape = probe.shape
        self.cacheType = probe.scalarType
        self.precision = probe.scalarType == .float16 ? "fp16" : "fp32"
        self.maxContext = probe.shape[3]
        (self.keyCache, self.valueCache) = Self.zeroCaches(shape: probe.shape, type: probe.scalarType)
    }

    /// Zeroed, not just allocated: masked slots get softmax weight 0, but
    /// 0 × NaN from uninitialised memory would still poison attn · v.
    static func zeroCaches(shape: [Int], type: NDArray.ScalarType) -> (NDArray, NDArray) {
        let count = shape.reduce(1, *)
        func zeros() -> NDArray {
            type == .float16
                ? NDArray(scalars: [Float16](repeating: 0, count: count), shape: shape)
                : NDArray(scalars: [Float](repeating: 0, count: count), shape: shape)
        }
        return (zeros(), zeros())
    }

    func reset() {
        (keyCache, valueCache) = Self.zeroCaches(shape: cacheShape, type: cacheType)
        position = 0
    }

    /// Append `ids` at the current position; returns the greedy next token.
    func prefill(_ ids: [Int]) async throws -> Int {
        var next = -1
        for start in stride(from: 0, to: ids.count, by: prefillLength) {
            let chunk = Array(ids[start..<min(start + prefillLength, ids.count)])
            // Left-pad to the static length. Pad rows point at the last slot,
            // which nothing real attends to before a decode step overwrites it
            // (llm_ai.model.left_pad); so the prompt must leave that slot free.
            guard position + chunk.count < maxContext else { throw GeneratorError.contextExhausted }
            let pad = prefillLength - chunk.count
            let tokens = [Int32](repeating: Int32(padID), count: pad) + chunk.map(Int32.init)
            let positions = [Int32](repeating: Int32(maxContext - 1), count: pad)
                + (position..<(position + chunk.count)).map(Int32.init)
            next = try await run(prefillFunction, tokens: tokens, positions: positions)
            position += chunk.count
        }
        return next
    }

    func decode(_ token: Int) async throws -> Int {
        guard position < maxContext else { throw GeneratorError.contextExhausted }
        let next = try await run(decodeFunction, tokens: [Int32(token)], positions: [Int32(position)])
        position += 1
        return next
    }

    private func run(_ function: InferenceFunction, tokens: [Int32], positions: [Int32]) async throws -> Int {
        // input_ids / position_ids are int32 in the converted model (gotcha 4).
        let inputs = [
            "input_ids": NDArray(scalars: tokens, shape: [1, tokens.count]),
            "position_ids": NDArray(scalars: positions, shape: [1, positions.count]),
        ]
        // The views borrow both caches at once, which exclusivity forbids on
        // two stored properties, so they move into locals for the call. Moved,
        // not copied: `var keys = keyCache` would leave two references to the
        // storage and taking the mutable view would copy both 21 MB caches
        // every token (~3 ms per decode step; gotcha 19).
        guard var keys = keyCache.take(), var values = valueCache.take() else {
            throw GeneratorError.missingState("keyCache/valueCache (call in progress)")
        }
        defer {
            keyCache = keys
            valueCache = values
        }
        var states = InferenceFunction.MutableViews()
        states.insert(&keys, for: "keyCache")
        states.insert(&values, for: "valueCache")
        var outputs = try await function.run(inputs: inputs, states: states)
        guard let logits = outputs.remove("logits")?.ndArray else { throw GeneratorError.missingOutput("logits") }
        return argmaxLastToken(logits)
    }
}

/// Greedy pick from `logits[1, T, vocab]` (the newest token). Reads through
/// the strides: the runtime may hand back a non-contiguous buffer.
@available(macOS 27, iOS 27, *)
func argmaxLastToken(_ logits: NDArray) -> Int {
    func scan<T: BinaryFloatingPoint & BitwiseCopyable>(_ type: T.Type) -> Int {
        logits.view(as: T.self).withUnsafePointer { p, shape, strides in
            let base = (shape[1] - 1) * strides[1], step = strides[2]
            var best = 0, bestValue = p[base]
            for i in 1..<shape[2] where p[base + i * step] > bestValue {
                best = i
                bestValue = p[base + i * step]
            }
            return best
        }
    }
    return logits.scalarType == .float16 ? scan(Float16.self) : scan(Float.self)
}
#endif

/// True when this binary was compiled against an SDK that ships CoreAI.framework.
public let hasCoreAIFramework: Bool = {
    #if canImport(CoreAI)
    return true
    #else
    return false
    #endif
}()
