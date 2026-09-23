import Foundation

// In-process `Decider` bench on CoreAI.framework: the `CoreAI.framework`
// column of docs/bench/README.md. Inputs are pre-tokenized by
// `python -m decide_ai.bench_ids` (no Swift BPE until step 2), so timings are
// the model call only, like `infer_ms` in `decide_ai.bench --backend local`.

/// `data/decide/bench_ids.json`: unpadded ids per L, per holdout state, per bench question.
public struct BenchIDs: Decodable, Sendable {
    public let padID: Int32
    public let questions: [String]
    public let stateCount: Int
    public let ids: [String: [[[Int32]]]]
    public let reference: Reference

    public struct Reference: Decodable, Sendable {
        public let state: Int
        public let L: Int
        public let logits: [[Float]]
    }

    enum CodingKeys: String, CodingKey {
        case padID = "pad_id", questions, stateCount = "state_count", ids, reference
    }

    public static func load(root: URL) throws -> BenchIDs {
        try JSONDecoder().decode(BenchIDs.self, from: Data(contentsOf: root.appending(path: "data/decide/bench_ids.json")))
    }

    /// `[n][L]` ids and mask for holdout state `s` and the first `n` questions, right-padded with `pad_id`.
    public func batch(state s: Int, n: Int, L: Int) -> (ids: [[Int32]], mask: [[Int32]]) {
        let rows = ids[String(L)]![s % stateCount].prefix(n)
        return (rows.map { $0 + Array(repeating: padID, count: L - $0.count) },
                rows.map { Array(repeating: 1, count: $0.count) + Array(repeating: 0, count: L - $0.count) })
    }
}

public struct LoadMs: Encodable, Sendable {
    public let first: Double
    public let restMean: Double
    public let all: [Double]
    enum CodingKeys: String, CodingKey { case first, all, restMean = "rest_mean" }
}

/// One `{asset, N, L}` row, shaped like `decide_ai.bench --backend local` rows.
public struct LocalBenchRow: Encodable, Sendable {
    public var asset: String
    public var variant: String
    public var function: String
    public var N: Int
    public var L: Int
    public var loadMs: LoadMs
    /// `model.loadFunction(named:)` for this row's function.
    public var functionLoadMs: Double
    public var firstCallMs: Double
    public var inferMs: BenchStats

    enum CodingKeys: String, CodingKey {
        case asset, variant, function, N, L
        case loadMs = "load_ms", functionLoadMs = "function_load_ms"
        case firstCallMs = "first_call_ms", inferMs = "infer_ms"
    }
}

func ms(_ d: Duration) -> Double {
    Double(d.components.seconds) * 1e3 + Double(d.components.attoseconds) / 1e15
}

#if canImport(CoreAI)
import CoreAI

public struct LocalBenchConfig: Sendable {
    public var batches = [1, 4, 8, 16]
    public var lengths = [64, 128]
    public var warmup = 20
    public var calls = 200
    public var minCalls = 30
    public var budgetS = 60.0
    public var maxRefDiff: Float = 1e-3
    public init() {}
}

/// Loads `modelURL` three times (like bench.py), checks the reference logits,
/// then times every N x L row. The static asset is detected by its `main_n{N}_l{L}` functions.
@available(macOS 27, iOS 27, *)
public func runLocalBench(modelURL: URL, inputs: BenchIDs, config c: LocalBenchConfig,
                          log: (String) -> Void) async throws -> (rows: [LocalBenchRow], refMaxDiff: Float) {
    let clock = ContinuousClock()
    var loads: [Double] = []
    var model: AIModel?
    for _ in 0..<3 {
        let t0 = clock.now
        model = try await AIModel(contentsOf: modelURL)
        loads.append(ms(clock.now - t0))
    }
    let m = model!
    let isStatic = m.functionNames.contains { $0.hasPrefix("main_n") }
    let variant = isStatic ? "static" : "dynamic"
    let loadMs = LoadMs(first: loads[0], restMean: loads.dropFirst().reduce(0, +) / Double(loads.count - 1), all: loads)
    log(String(format: "%@ (%@): load ms first %.0f, rest %.0f; functions %@", modelURL.lastPathComponent, variant,
               loadMs.first, loadMs.restMean, m.functionNames.sorted().joined(separator: " ")))
    func name(_ n: Int, _ l: Int) -> String { isStatic ? "main_n\(n)_l\(l)" : "main" }

    // Same numbers as the Python Core AI path before anything is timed.
    let ref = inputs.reference
    let (rIDs, rMask) = inputs.batch(state: ref.state, n: ref.logits.count, L: ref.L)
    let got = try await Decider(model: m, functionName: name(ref.logits.count, ref.L)).logits(inputIDs: rIDs, attentionMask: rMask)
    let diff = zip(got, ref.logits).flatMap { zip($0, $1).map { abs($0 - $1) } }.max() ?? .infinity
    log(String(format: "  reference check (state %d, N=%d, L=%d): max |diff| vs Python = %.2e", ref.state, ref.logits.count, ref.L, diff))
    guard diff <= c.maxRefDiff else { throw DeciderError.referenceMismatch(Double(diff)) }

    var rows: [LocalBenchRow] = []
    for n in c.batches {
        for l in c.lengths {
            let fn = name(n, l)
            let t0 = clock.now
            let decider = try Decider(model: m, functionName: fn)
            let fnLoad = ms(clock.now - t0)
            let batches = (0..<inputs.stateCount).map { inputs.batch(state: $0, n: n, L: l) }
            func call(_ i: Int) async throws -> Double {
                let b = batches[i % batches.count]
                let t = clock.now
                _ = try await decider.logits(inputIDs: b.ids, attentionMask: b.mask)
                return ms(clock.now - t)
            }
            let first = try await call(0)
            for i in 0..<c.warmup { _ = try await call(i) }
            var xs: [Double] = []
            let start = clock.now
            while xs.count < c.calls && (xs.count < c.minCalls || ms(clock.now - start) < c.budgetS * 1e3) {
                xs.append(try await call(xs.count))
            }
            let row = LocalBenchRow(asset: modelURL.lastPathComponent, variant: variant, function: fn, N: n, L: l,
                                    loadMs: loadMs, functionLoadMs: fnLoad, firstCallMs: first, inferMs: BenchStats(xs))
            rows.append(row)
            log(String(format: "  %@ N=%2d L=%3d %-14@ infer p50 %8.2f p95 %8.2f ms  first %.1f  fn load %.1f (n=%d)",
                       variant, n, l, fn, row.inferMs.p50, row.inferMs.p95, first, fnLoad, xs.count))
        }
    }
    return (rows, diff)
}
#endif
