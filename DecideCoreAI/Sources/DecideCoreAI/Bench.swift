import Foundation

// `decide-bench/1` records: the same shape `python -m decide_ai.bench --json`
// writes, so Python in-process, Python->HTTP, Swift->HTTP and (on macOS 27)
// the in-process `Decider` can be compared from `docs/bench/`.

public struct BenchStats: Encodable, Sendable {
    public let count: Int
    public let mean: Double
    public let p50: Double
    public let p95: Double
    public let min: Double
    public let max: Double

    public init(_ xs: [Double]) {
        let sorted = xs.sorted()
        count = xs.count
        mean = xs.isEmpty ? .nan : xs.reduce(0, +) / Double(xs.count)
        func pct(_ p: Double) -> Double {
            guard !sorted.isEmpty else { return .nan }
            // numpy's linear interpolation, so the numbers line up with bench.py
            let pos = p * Double(sorted.count - 1)
            let lo = Int(pos.rounded(.down)), hi = Swift.min(lo + 1, sorted.count - 1)
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - Double(lo))
        }
        p50 = pct(0.5)
        p95 = pct(0.95)
        min = sorted.first ?? .nan
        max = sorted.last ?? .nan
    }
}

public struct BenchRow: Encodable, Sendable {
    public var asset: String
    public var N: Int
    /// nil for the Laya backend (no padded length to pin; encodes as JSON null like bench.py's rows).
    public var L: Int?
    public var firstCallMs: Double
    public var httpOverheadP50Ms: Double
    public var roundtripMs: BenchStats
    /// nil for the Laya backend, whose server reports no tokenize / infer split.
    public var serverInferMs: BenchStats?
    public var serverTotalMs: BenchStats

    enum CodingKeys: String, CodingKey {
        case asset, N, L
        case firstCallMs = "first_call_ms"
        case httpOverheadP50Ms = "http_overhead_p50_ms"
        case roundtripMs = "roundtrip_ms"
        case serverInferMs = "server_infer_ms"
        case serverTotalMs = "server_total_ms"
    }

    public init(asset: String, N: Int, L: Int?, firstCallMs: Double, httpOverheadP50Ms: Double,
                roundtripMs: BenchStats, serverInferMs: BenchStats?, serverTotalMs: BenchStats) {
        self.asset = asset; self.N = N; self.L = L
        self.firstCallMs = firstCallMs; self.httpOverheadP50Ms = httpOverheadP50Ms
        self.roundtripMs = roundtripMs; self.serverInferMs = serverInferMs; self.serverTotalMs = serverTotalMs
    }

    // `L` is written as an explicit null (JSONEncoder would drop the key), matching bench.py's Laya rows;
    // `server_infer_ms` is simply absent when the server reports no split.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(asset, forKey: .asset)
        try c.encode(N, forKey: .N)
        if let L { try c.encode(L, forKey: .L) } else { try c.encodeNil(forKey: .L) }
        try c.encode(firstCallMs, forKey: .firstCallMs)
        try c.encode(httpOverheadP50Ms, forKey: .httpOverheadP50Ms)
        try c.encode(roundtripMs, forKey: .roundtripMs)
        try c.encodeIfPresent(serverInferMs, forKey: .serverInferMs)
        try c.encode(serverTotalMs, forKey: .serverTotalMs)
    }
}

public struct BenchRecord: Encodable, Sendable {
    public let schema = "decide-bench/1"
    public var runtime: String
    public var recordedAt: String
    public var host: [String: String]
    public var config: [String: String]
    public var rows: [BenchRow]

    enum CodingKeys: String, CodingKey { case schema, runtime, host, config, rows, recordedAt = "recorded_at" }

    public init(runtime: String, config: [String: String], rows: [BenchRow]) {
        self.runtime = runtime
        self.recordedAt = ISO8601DateFormatter().string(from: Date())
        self.host = [
            "machine": Self.sysctl("hw.machine"),
            "chip": Self.sysctl("machdep.cpu.brand_string"),
            "macos": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        self.config = config
        self.rows = rows
    }

    public func write(to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (try enc.encode(self) + Data("\n".utf8)).write(to: url)
    }

    private static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
        return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

/// Repo root found by walking up from the working directory (looks for `docs/`).
public func repoRoot() -> URL? {
    var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    for _ in 0..<4 {
        if FileManager.default.fileExists(atPath: dir.appending(path: "docs").path) { return dir }
        dir = dir.deletingLastPathComponent()
    }
    return nil
}

/// The same inputs `decide_ai.bench` uses: holdout states and the 16 bench questions.
public func benchInputs(root: URL) throws -> (states: [String], questions: [(String, Question)]) {
    let holdout = try String(contentsOf: root.appending(path: "data/decide/holdout.jsonl"), encoding: .utf8)
    struct Row: Decodable { let state: String }
    let states = try holdout.split(separator: "\n").filter { !$0.isEmpty }
        .map { try JSONDecoder().decode(Row.self, from: Data($0.utf8)).state }
    let qdata = try Data(contentsOf: root.appending(path: "data/decide/bench_questions.json"))
    // Keep file order (JSON objects are unordered in Codable): parse keys by hand.
    let text = String(decoding: qdata, as: UTF8.self)
    let dict = try JSONDecoder().decode([String: Question].self, from: qdata)
    var ordered: [(String, Question)] = []
    for line in text.split(separator: "\n") {
        guard let q = line.firstIndex(of: "\""), let e = line[line.index(after: q)...].firstIndex(of: "\"") else { continue }
        let key = String(line[line.index(after: q)..<e])
        if let v = dict[key], !ordered.contains(where: { $0.0 == key }) { ordered.append((key, v)) }
    }
    return (states, ordered)
}
