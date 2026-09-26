import Foundation
import LLMCoreAI
#if canImport(CoreAI)
import CoreAI
#endif

// llm-cli                                   which generators this build can use
// llm-cli --model ../models/llm/SmolLM2Stateful.aimodel [--chat 0 | --bench --json out.json]
//         [--tokenizer ../models/llm/hf/SmolLM2-360M-Instruct/tokenizer.json]
//         [--prompts ../data/llm/prompt_ids.json] [--compute gpu|default] [--cold]
//         [--tokens 128] [--runs 5]
//
// With --model it always checks itself first: greedy tokens for the reference
// prompt must match the PyTorch fp32 run stored in prompt_ids.json. --cold
// deletes this program's specialization cache entry for the asset before
// loading, so the load time is a real first load.

struct Args {
    var values: [String: String] = [:]
    var flags: Set<String> = []
    init(_ argv: [String]) {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--"), i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
                values[String(a.dropFirst(2))] = argv[i + 1]
                i += 2
            } else {
                flags.insert(String(a.dropFirst(2)))
                i += 1
            }
        }
    }
    func string(_ k: String, _ d: String) -> String { values[k] ?? d }
    func int(_ k: String, _ d: Int) -> Int { values[k].flatMap(Int.init) ?? d }
}

let args = Args(Array(CommandLine.arguments.dropFirst()))
setvbuf(stdout, nil, _IOLBF, 0)  // line-buffered even when redirected, so the log reads in order

guard let modelPath = args.values["model"] else {
    print("CoreAI.framework in SDK: \(hasCoreAIFramework ? "yes" : "no (needs Xcode 27 / macOS 27)")")
    do {
        let remote = try await RemoteGenerator()
        print("remote server: \(remote.label), max_seq_len=\(remote.info.maxSeqLen)")
    } catch {
        print("remote server: not reachable on :8770 — start `python -m llm_ai.serve` (\(error))")
    }
    print("in-process: llm-cli --model ../models/llm/SmolLM2Stateful.aimodel [--chat 0] [--bench --json FILE]")
    exit(0)
}

#if canImport(CoreAI)
if #available(macOS 27, *) {
    try await runModel()
} else {
    print("CoreAI.framework needs macOS 27")
    exit(1)
}

@available(macOS 27, *)
func runModel() async throws {
    let modelURL = URL(fileURLWithPath: modelPath)
    let prompts = try PromptIDs.load(URL(fileURLWithPath: args.string("prompts", "../data/llm/prompt_ids.json")))
    let tokenizerURL = URL(fileURLWithPath: args.string("tokenizer", "../models/llm/hf/SmolLM2-360M-Instruct/tokenizer.json"))
    let compute = args.string("compute", "gpu")
    let options: SpecializationOptions = compute == "default" ? .default : ModelGenerator.defaultOptions

    let cold = args.flags.contains("cold")
    if cold {
        try AIModelCache.default.deleteEntry(for: modelURL, options: options)
    }
    // Not probed with AIModelCache.model(for:): that loads the model itself,
    // and a second 0.7 GB load right after it ran 13 s instead of ~0.1 s.
    let generator = try await ModelGenerator(modelURL: modelURL, tokenizerURL: tokenizerURL, prompts: prompts, options: options)
    let loadCache = cold ? "cold" : "cached"
    print("\(generator.label)\nloaded in \(String(format: "%.0f", generator.loadMS)) ms (\(loadCache), compute=\(compute), \(generator.precision))")

    // Self-check against the PyTorch reference before timing anything.
    let ref = prompts.reference
    var got: [Int] = []
    for try await t in generator.generate(promptIDs: prompts.chat[ref.chatIndex].ids, maxTokens: ref.greedyIds.count, stopAtEOS: false) {
        got.append(t.id)
    }
    let agree = zip(got, ref.greedyIds).filter { $0 == $1 }.count
    let firstDiff = zip(got, ref.greedyIds).enumerated().first { $0.element.0 != $0.element.1 }?.offset
    print("reference check: \(agree)/\(ref.greedyIds.count) greedy tokens match PyTorch fp32"
          + (firstDiff.map { ", first difference at step \($0)" } ?? ""))
    guard agree >= ref.greedyIds.count - 1 else {
        print("  got:      \(generator.decoder.decode(got).debugDescription)\n  expected: \(ref.text.debugDescription)")
        exit(1)
    }

    if let chatIndex = args.values["chat"].flatMap(Int.init) {
        try await generator.reset()
        let chat = prompts.chat[chatIndex]
        print("\n> \(chat.prompt)\n")
        var count = 0, decodeMS = 0.0, firstMS = 0.0
        for try await t in generator.generate(promptIDs: chat.ids, maxTokens: args.int("tokens", 256)) {
            print(t.text, terminator: "")
            fflush(stdout)
            if count == 0 { firstMS = t.ms } else { decodeMS += t.ms }
            count += 1
        }
        print(String(format: "\n\n[%d prompt tokens, first token %.0f ms, %d tokens, %.1f tok/s decode]",
                     chat.ids.count, firstMS, count, Double(count - 1) / decodeMS * 1e3))
    }

    if args.flags.contains("bench") {
        let tokens = args.int("tokens", 128), runs = args.int("runs", 5)
        // Warm-up: the first calls pay one-off setup.
        for try await _ in generator.generate(promptIDs: prompts.bench["16"]!, maxTokens: 4, stopAtEOS: false) {}
        var records: [BenchRecord] = []
        for n in [16, 128, 512] {
            guard let ids = prompts.bench[String(n)] else { continue }
            var prefills: [Double] = [], steps: [Double] = [], texts = Set<[Int]>()
            for _ in 0..<runs {
                try await generator.reset()
                var out: [Int] = []
                for try await t in generator.generate(promptIDs: ids, maxTokens: tokens, stopAtEOS: false) {
                    if out.isEmpty { prefills.append(t.ms) } else { steps.append(t.ms) }
                    out.append(t.id)
                }
                texts.insert(out)
            }
            let dec = Stats(steps), pre = Stats(prefills)
            print(String(format: "  T=%3d: prefill p50 %.0f ms, decode p50 %.1f / p95 %.1f ms, %.2f tok/s",
                         n, pre.p50, dec.p50, dec.p95, 1e3 / dec.mean))
            records.append(BenchRecord(
                variant: "stateful", runtime: "framework", compute: compute, precision: generator.precision,
                prompt_tokens: n, generated_tokens: tokens, runs: runs,
                load_ms: .init(value: round3(generator.loadMS), cache: loadCache),
                prefill_ms: pre, decode_ms: dec, tok_per_s: round3(1e3 / dec.mean),
                deterministic: texts.count == 1, reference_check: "\(agree)/\(ref.greedyIds.count)",
                peak_rss_mb: 0, load_avg: [], host: hostInfo(), git: gitSHA(),
                recorded_at: ISO8601DateFormatter().string(from: Date()), asset: modelPath, build: buildConfiguration))
        }
        let peak = round3(peakRSSMB()), load = loadAverage()
        for i in records.indices {
            records[i].peak_rss_mb = peak
            records[i].load_avg = load
        }
        if let path = args.values["json"] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(BenchFile(schema: "llm-bench/1", records: records)).write(to: URL(fileURLWithPath: path))
            print("wrote \(path)")
        }
    }
}
#else
print("this build has no CoreAI.framework (needs Xcode 27 / macOS 27); use the remote generator")
exit(1)
#endif

// MARK: - bench helpers

// Computed, not a stored top-level `let`: globals in main.swift are
// initialized in source order, and runModel() runs before this line.
nonisolated var buildConfiguration: String {
    #if DEBUG
    return "debug"
    #else
    return "release"
    #endif
}

func round3(_ x: Double) -> Double { (x * 1000).rounded() / 1000 }

struct Stats: Encodable {
    var count = 0, mean = 0.0, p50 = 0.0, p95 = 0.0, min = 0.0
    init(_ xs: [Double]) {
        guard !xs.isEmpty else { return }
        let s = xs.sorted()
        func pct(_ p: Double) -> Double {
            let r = p / 100 * Double(s.count - 1), lo = Int(r), hi = Swift.min(lo + 1, s.count - 1)
            return s[lo] + (s[hi] - s[lo]) * (r - Double(lo))
        }
        count = s.count
        mean = round3(s.reduce(0, +) / Double(s.count))
        p50 = round3(pct(50))
        p95 = round3(pct(95))
        min = round3(s[0])
    }
}

/// One `llm-bench/1` record; keys match `llm_ai.play` so the two files diff.
struct BenchRecord: Encodable {
    struct Load: Encodable { var value: Double; var cache: String }
    var variant, runtime, compute, precision: String
    var prompt_tokens, generated_tokens, runs: Int
    var load_ms: Load
    var prefill_ms, decode_ms: Stats
    var tok_per_s: Double
    var deterministic: Bool
    var reference_check: String
    var peak_rss_mb: Double
    var load_avg: [Double]
    var host: [String: String]
    var git, recorded_at, asset, build: String
}

struct BenchFile: Encodable {
    var schema: String
    var records: [BenchRecord]
}

func peakRSSMB() -> Double {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    return Double(usage.ru_maxrss) / 1e6  // bytes on macOS
}

func loadAverage() -> [Double] {
    var l = [Double](repeating: 0, count: 3)
    getloadavg(&l, 3)
    return l.map { ($0 * 100).rounded() / 100 }
}

func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    var buf = [UInt8](repeating: 0, count: size)
    sysctlbyname(name, &buf, &size, nil, 0)
    return String(decoding: buf.prefix { $0 != 0 }, as: UTF8.self)
}

func hostInfo() -> [String: String] {
    ["chip": sysctlString("machdep.cpu.brand_string"), "machine": "arm64",
     "macos": ProcessInfo.processInfo.operatingSystemVersionString]
}

func gitSHA() -> String {
    func git(_ a: [String]) -> String {
        let p = Process(), pipe = Pipe()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        p.arguments = a
        p.standardOutput = pipe
        try? p.run()
        p.waitUntilExit()
        return String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    let dirty = !git(["status", "--porcelain", "--untracked-files=no"]).isEmpty
    return git(["rev-parse", "--short", "HEAD"]) + (dirty ? "-dirty" : "")
}
