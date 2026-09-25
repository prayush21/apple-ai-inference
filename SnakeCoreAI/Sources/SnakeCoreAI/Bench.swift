import Foundation

// `snake-bench/1` records: the same shape `python -m snake_ai.play --json`
// writes, so the Python runtime, the remote player, and (on macOS 27) the
// in-process `ModelPlayer` can be compared from `docs/bench/`.

public struct BenchStats: Encodable, Sendable {
    public let count: Int
    public let mean: Double?
    public let p50: Double?
    public let p95: Double?
    public let first5: Double?
    /// Mean of the last 5 moves, only when there are at least 20 (see play.py).
    public let last5: Double?

    enum CodingKeys: String, CodingKey {
        case count, mean, p50, p95
        case first5 = "first_5"
        case last5 = "last_5"
    }

    public init(_ xs: [Double]) {
        count = xs.count
        guard !xs.isEmpty else { mean = nil; p50 = nil; p95 = nil; first5 = nil; last5 = nil; return }
        let sorted = xs.sorted()
        func pct(_ p: Double) -> Double { sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * p))] }
        func avg(_ s: ArraySlice<Double>) -> Double { s.reduce(0, +) / Double(s.count) }
        mean = avg(xs[...])
        p50 = pct(0.5)
        p95 = pct(0.95)
        first5 = avg(xs.prefix(5))
        last5 = xs.count >= 20 ? avg(xs.suffix(5)) : nil
    }
}

public struct BenchLoad: Encodable, Sendable {
    public let first: Double
    public let restMean: Double?
    public let all: [Double]

    enum CodingKeys: String, CodingKey { case first, all, restMean = "rest_mean" }

    public init(_ loads: [Double]) {
        first = loads.first ?? 0
        restMean = loads.count > 1 ? loads.dropFirst().reduce(0, +) / Double(loads.count - 1) : nil
        all = loads
    }
}

public struct BenchPlayer: Encodable, Sendable {
    public var player: String
    public var label: String
    public var games: Int
    public var wins: Int
    public var draws: Int
    public var avgSteps: Double
    public var loadMs: BenchLoad
    /// Wall-clock `chooseAction` per move, as the caller experiences it.
    public var inferenceMs: BenchStats
    /// `RemoteModelPlayer` only: the server's own inference time per move.
    public var serverInferenceMs: BenchStats?

    enum CodingKeys: String, CodingKey {
        case player, label, games, wins, draws
        case avgSteps = "avg_steps"
        case loadMs = "load_ms"
        case inferenceMs = "inference_ms"
        case serverInferenceMs = "server_inference_ms"
    }

    public init(player: String, label: String, games: Int, wins: Int, draws: Int, avgSteps: Double,
                loads: [Double], inference: [Double], serverInference: [Double]?) {
        self.player = player
        self.label = label
        self.games = games
        self.wins = wins
        self.draws = draws
        self.avgSteps = avgSteps
        self.loadMs = BenchLoad(loads)
        self.inferenceMs = BenchStats(inference)
        self.serverInferenceMs = serverInference.map(BenchStats.init)
    }
}

/// Minimal JSON value for the free-form `config` block.
public enum BenchValue: Encodable, Sendable, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
                        ExpressibleByBooleanLiteral, ExpressibleByStringLiteral {
    case int(Int), double(Double), bool(Bool), string(String)

    public init(integerLiteral v: Int) { self = .int(v) }
    public init(floatLiteral v: Double) { self = .double(v) }
    public init(booleanLiteral v: Bool) { self = .bool(v) }
    public init(stringLiteral v: String) { self = .string(v) }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        }
    }
}

public struct BenchRecord: Encodable, Sendable {
    public let schema = "snake-bench/1"
    public var runtime: String
    public var recordedAt: String
    public var host: [String: String]
    public var config: [String: BenchValue]
    public var players: [BenchPlayer]

    enum CodingKeys: String, CodingKey {
        case schema, runtime, host, config, players
        case recordedAt = "recorded_at"
    }

    public init(runtime: String, config: [String: BenchValue], players: [BenchPlayer]) {
        self.runtime = runtime
        self.recordedAt = ISO8601DateFormatter().string(from: Date())
        self.host = [
            "machine": Self.sysctl("hw.machine"),
            "chip": Self.sysctl("machdep.cpu.brand_string"),
            "macos": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        self.config = config
        self.players = players
    }

    /// Which Core AI path a player actually took, for the `runtime` field.
    public static func runtimeDescription(for player: any SnakePlayer) -> String {
        if player is RemoteModelPlayer { return "coreai.runtime via snake_ai.serve (HTTP from Swift)" }
        if #available(macOS 27, iOS 27, *), player is ModelPlayer || player is StatelessModelPlayer {
            return "CoreAI.framework (Swift, in-process)"
        }
        return "no model (\(type(of: player)))"
    }

    public func write(to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (try enc.encode(self) + Data("\n".utf8)).write(to: url)
    }

    /// `docs/bench/` in the repo when run from inside it (the same upward
    /// search as the app's model discovery), else `~/Downloads`.
    public static func defaultDirectory() -> URL {
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<4 {
            let docs = dir.appending(path: "docs")
            if FileManager.default.fileExists(atPath: docs.path) { return docs.appending(path: "bench") }
            dir = dir.deletingLastPathComponent()
        }
        return FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
    }

    private static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
        return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}
