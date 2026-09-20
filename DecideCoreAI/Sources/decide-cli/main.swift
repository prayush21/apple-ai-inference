import DecideCoreAI
import Foundation

// decide-cli: one decision, or the latency matrix through decide_ai.serve.
//
//   swift run decide-cli                                   # 5 triage questions on a sample state
//   swift run decide-cli --state "Order #48213 never showed up."
//   swift run decide-cli --bench --json ../docs/bench/decide-swift-remote.json
//   swift run decide-cli --bench --json ../docs/bench/decide-swift-remote-laya.json   # server: serve --backend laya
//
// On macOS 27 `--model models/decide/NLICrossEncoderStatic.aimodel` would use
// the in-process `Decider`; there is no tokenizer in Swift yet (step 2), so
// the CLI only offers the remote path today.

struct Args {
    var rest = Array(CommandLine.arguments.dropFirst())
    mutating func flag(_ name: String) -> Bool {
        if let i = rest.firstIndex(of: name) { rest.remove(at: i); return true }
        return false
    }
    mutating func option(_ name: String) -> String? {
        guard let i = rest.firstIndex(of: name), i + 1 < rest.count else { return nil }
        let v = rest[i + 1]; rest.removeSubrange(i...i + 1); return v
    }
}
var a = Args()
let bench = a.flag("--bench")
let jsonPath = a.option("--json")
let url = URL(string: a.option("--url") ?? "http://127.0.0.1:8770")!
let state = a.option("--state") ?? "The pasta was cold and the waiter ignored us. I want my money back."
let calls = Int(a.option("--calls") ?? "200")!
let warmup = Int(a.option("--warmup") ?? "20")!
let minCalls = Int(a.option("--min-calls") ?? "30")!
let budgetS = Double(a.option("--budget-s") ?? "60")!

let decider: RemoteDecider
do {
    decider = try await RemoteDecider(baseURL: url)
} catch {
    print("cannot reach decide_ai.serve at \(url): \(error)\nstart it with: .venv/bin/python -m decide_ai.serve")
    exit(1)
}
print("server: \(decider.info.backend) \(decider.info.asset) max_len=\(decider.info.maxLen) T=\(decider.info.temperature)")

if !bench {
    let r = try await decider.decide(state: state, questions: TriageQuestions.all)
    print("state: \(state)")
    for (name, a) in r.response.answers.sorted(by: { $0.key < $1.key }) {
        print("  " + name.padding(toLength: 24, withPad: " ", startingAt: 0) + String(format: " %.3f", a.probability))
    }
    let l = r.response.timing?.paddedLen.map { "L=\($0)" } ?? "no padded L"
    print(String(format: "round trip %.1f ms, server infer %.1f ms, total %.1f ms (batch %d, ",
                 r.roundtripMs, r.serverInferMs ?? .nan, r.serverTotalMs ?? .nan, r.response.timing?.batch ?? 0) + l + ")")
    exit(0)
}

guard let root = repoRoot() else { print("run from inside the repo (needs docs/ and data/decide/)"); exit(1) }
let (states, questions) = try benchInputs(root: root)
var rows: [BenchRow] = []
let laya = !decider.info.pinsPaddedLen
// Laya pads to its own sequence: run N only, L is nil, and there is no server infer split.
let lengths: [Int?] = laya ? [nil] : [64, 128].filter { $0 <= decider.info.maxLen }
if laya && warmup == 20 { print("laya: using 5 warm-ups (calls are ~0.3-1 s)") }
let warmups = laya && warmup == 20 ? 5 : warmup
for n in [1, 4, 8, 16] {
    for l in lengths {
        let qs = Dictionary(uniqueKeysWithValues: questions.prefix(n).map { ($0.0, $0.1) })
        func call(_ i: Int) async throws -> RemoteDecider.Result {
            try await decider.decide(state: states[i % states.count], questions: qs, paddedLen: l)
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        _ = try await call(0)
        let first = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        for i in 0..<warmups { _ = try await call(i) }
        var rt: [Double] = [], infer: [Double] = [], total: [Double] = []
        let start = Date()
        var k = 0
        while k < calls && (k < minCalls || Date().timeIntervalSince(start) < budgetS) {
            let r = try await call(k)
            rt.append(r.roundtripMs); total.append(r.serverTotalMs ?? .nan)
            if let ms = r.serverInferMs { infer.append(ms) }
            k += 1
        }
        let row = BenchRow(asset: decider.info.asset, N: n, L: l, firstCallMs: first,
                           httpOverheadP50Ms: BenchStats(rt).p50 - BenchStats(total).p50,
                           roundtripMs: BenchStats(rt), serverInferMs: laya ? nil : BenchStats(infer),
                           serverTotalMs: BenchStats(total))
        rows.append(row)
        let lText = l.map { String(format: "%3d", $0) } ?? "  —"
        print("  N=\(String(format: "%2d", n)) L=\(lText)" + String(
            format: " roundtrip p50 %8.1f ms, server total p50 %8.1f, HTTP+JSON overhead %.1f ms (n=%d)",
            row.roundtripMs.p50, row.serverTotalMs.p50, row.httpOverheadP50Ms, k))
    }
}
if let jsonPath {
    let record = BenchRecord(
        runtime: laya ? "convaiinnovations/laya (PyTorch, CPU) via decide_ai.serve --backend laya (HTTP from Swift)"
                      : "coreai.runtime via decide_ai.serve (HTTP from Swift)",
        config: ["backend": laya ? "swift-remote-laya" : "swift-remote", "url": url.absoluteString,
                 "warmup": "\(warmups)", "calls": "\(calls)", "min_calls": "\(minCalls)", "budget_s": "\(budgetS)",
                 "inputs": "data/decide/holdout.jsonl states x data/decide/bench_questions.json",
                 "note": laya ? "Laya pads to its own sequence, so N only and L is null; the server reports no infer split"
                              : "count per row is capped by budget_s; same matrix and inputs as decide_ai.bench"],
        rows: rows)
    let out = URL(fileURLWithPath: jsonPath, relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    try record.write(to: out)
    print("wrote \(out.path)")
}
