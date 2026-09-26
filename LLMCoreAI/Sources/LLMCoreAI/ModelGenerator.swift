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
/// `chat(_:)` / `generate(prompt:)` keep a multi-turn conversation in the
/// cache: text is encoded in Swift (`BPEEncoder`) with SmolLM2's chat
/// template, and each turn only prefills its own tokens. `generate(promptIDs:)`
/// appends raw ids (the bench). `reset()` starts over.
@available(macOS 27, iOS 27, *)
public final class ModelGenerator: TokenGenerator {
    public let label: String
    public let loadMS: Double
    public let decoder: ByteLevelDecoder
    public let encoder: BPEEncoder
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
        self.encoder = try BPEEncoder(tokenizerJSON: tokenizerURL)
        self.prompts = prompts
        self.eosID = prompts?.eosId ?? eosID
        self.maxContext = engine.maxContext
        self.precision = engine.precision
        let prefill = engine.prefillLength
        self.label = "CoreAI.framework (in-process) — \(modelURL.lastPathComponent), prefill t\(prefill) + decode"
    }

    /// Waits for a generation still running on the engine (e.g. one whose
    /// stream was just cancelled) before clearing the caches.
    public func reset() async throws {
        await engine.acquire()
        await engine.reset()
        await engine.release()
    }

    /// Tokens already in the cache.
    public var position: Int { get async { await engine.position } }

    public func generate(prompt: String, maxTokens: Int) -> AsyncThrowingStream<GeneratedToken, Error> {
        chat(prompt, maxTokens: maxTokens)
    }

    /// Send one user message in the ongoing conversation and stream the
    /// assistant's reply. The first message opens the chat template (with
    /// `system`, or SmolLM2's default system message); later ones continue it.
    public func chat(_ message: String, system: String? = nil, maxTokens: Int) -> AsyncThrowingStream<GeneratedToken, Error> {
        let engine = self.engine, encoder = self.encoder, eosID = self.eosID
        return stream(maxTokens: maxTokens, stopAtEOS: true, opensChat: true) {
            let (started, pending) = await engine.chatState()
            let text: String
            if started {
                // The previous reply's last token is still pending (not in the
                // cache). If it was <|im_end|> it is fed as-is; if the reply was
                // cut off, close the turn here.
                text = (pending == eosID ? "" : ChatTemplate.end) + "\n" + ChatTemplate.turn(user: message)
            } else {
                text = ChatTemplate.opening(system: system, user: message)
            }
            return encoder.encode(text)
        }
    }

    /// Append `promptIDs` to the conversation and stream up to `maxTokens`
    /// greedy tokens. `stopAtEOS: false` always produces `maxTokens` (bench).
    public func generate(promptIDs: [Int], maxTokens: Int, stopAtEOS: Bool = true) -> AsyncThrowingStream<GeneratedToken, Error> {
        stream(maxTokens: maxTokens, stopAtEOS: stopAtEOS) { promptIDs }
    }

    /// Prefill the ids `prompt` returns, then decode. Each token's `text` is
    /// the reply decoded so far minus what earlier tokens already produced,
    /// so multi-byte characters are never split. The first token's `ms` is
    /// the prefill (time to first token). Cancelling stops between tokens
    /// and leaves the cache consistent (the last token stays pending).
    ///
    /// Generations hold the engine for their whole run, so a cancelled one
    /// (its consumer is gone, but a prefill or decode call is still in
    /// flight) finishes before the next one reads `pending` and `position`.
    private func stream(maxTokens: Int, stopAtEOS: Bool, opensChat: Bool = false,
                        prompt: @escaping @Sendable () async throws -> [Int]) -> AsyncThrowingStream<GeneratedToken, Error> {
        let engine = self.engine, decoder = self.decoder, eosID = self.eosID
        return AsyncThrowingStream { continuation in
            let task = Task {
                await engine.acquire()
                guard !Task.isCancelled else {
                    await engine.release()
                    continuation.finish()
                    return
                }
                do {
                    var clock = ContinuousClock.now
                    var next = try await engine.prefill(try await prompt(), opensChat: opensChat)
                    // Emitted text is tracked in UTF-8 bytes, not Characters: a
                    // token that only adds a combining mark, ZWJ or emoji
                    // modifier leaves the Character count unchanged.
                    var reply: [Int] = [], emittedBytes = 0
                    for _ in 0..<maxTokens {
                        let ms = Self.ms(since: clock)
                        if stopAtEOS && next == eosID { break }
                        reply.append(next)
                        let text = decoder.decode(reply)
                        var piece = ""
                        if !text.hasSuffix("\u{FFFD}") {
                            piece = String(decoding: text.utf8.dropFirst(emittedBytes), as: UTF8.self)
                            emittedBytes = text.utf8.count
                        }
                        continuation.yield(GeneratedToken(id: next, text: piece, ms: ms))
                        if reply.count == maxTokens || Task.isCancelled { break }
                        clock = ContinuousClock.now
                        next = try await engine.decode(next)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                await engine.release()
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
    /// The last generated token, returned but not yet written to the cache.
    /// The next prefill or decode feeds it first.
    private(set) var pending: Int?
    private var chatStarted = false
    /// One generation at a time (`acquire` / `release`): actor methods are
    /// re-entrant at every `await`, so the actor alone does not serialize them.
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

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
        pending = nil
        chatStarted = false
    }

    func acquire() async {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
    }

    /// Hands the engine to the next waiter (still busy) or frees it.
    func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }

    /// Whether the chat template has been opened, and the pending token.
    func chatState() -> (started: Bool, pending: Int?) {
        (chatStarted, pending)
    }

    /// Append the pending token (if any) and `ids` at the current position;
    /// returns the greedy next token, which becomes pending. `opensChat`
    /// marks the chat as started, only once the prefill has succeeded. On
    /// failure `position` and `pending` roll back: slots past `position` are
    /// rewritten before anything attends to them, so the cache stays usable.
    func prefill(_ newIDs: [Int], opensChat: Bool = false) async throws -> Int {
        let ids = (pending.map { [$0] } ?? []) + newIDs
        guard position + ids.count < maxContext else { throw GeneratorError.contextExhausted }
        let startPosition = position
        var next = -1
        do {
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
        } catch {
            position = startPosition
            throw error
        }
        pending = next
        if opensChat { chatStarted = true }
        return next
    }

    /// Write `token` (the pending one) to the cache; returns the next.
    func decode(_ token: Int) async throws -> Int {
        guard position < maxContext else { throw GeneratorError.contextExhausted }
        let next = try await run(decodeFunction, tokens: [Int32(token)], positions: [Int32(position)])
        position += 1
        pending = next
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
