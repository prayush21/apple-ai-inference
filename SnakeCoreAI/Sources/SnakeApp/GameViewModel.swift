import Foundation
import Observation
import SnakeCoreAI
import SnakeEngine

/// Which policy drives snake A. Switchable from the UI so the three can be
/// compared head-to-head on the same board.
enum AIKind: String, CaseIterable, Identifiable {
    case heuristic = "Heuristic"
    case minimax = "Minimax"
    case model = "Model (heur)"
    case modelMinimaxTaught = "Model (minimax)"
    case modelMixedTaught = "Model (mixed)"

    var id: String { rawValue }

    var blurb: String {
        switch self {
        case .heuristic: "greedy flood-fill, one move ahead; treats the opponent as static"
        case .minimax: "4-ply adversarial search over Voronoi territory; the stronger teacher"
        case .model: "converted transformer trained to imitate the heuristic (Core AI runtime, :8765)"
        case .modelMinimaxTaught: "same transformer trained to imitate minimax (Core AI runtime, :8766)"
        case .modelMixedTaught: "same transformer trained on alternating heuristic/minimax games (Core AI runtime, :8767)"
        }
    }

    /// Port of the `python -m snake_ai.serve` instance for model kinds.
    var serverPort: Int? {
        switch self {
        case .model: 8765
        case .modelMinimaxTaught: 8766
        case .modelMixedTaught: 8767
        default: nil
        }
    }
}

struct Scoreboard: Equatable {
    var aiWins = 0, humanWins = 0, draws = 0
    var games: Int { aiWins + humanWins + draws }
}

/// Drives the game loop. Snake 0 ("A") is the AI player, snake 1 ("B") is the
/// human. The loop ticks on a fixed interval; the human's most recent arrow
/// key is applied at the next tick, mirroring how the talk's app feeds the
/// model one board state per step.
@MainActor
@Observable
final class GameViewModel {
    enum Phase: Equatable { case idle, loading, running, over(String) }

    private(set) var game = SnakeGame(seed: 1)
    private(set) var phase: Phase = .idle
    private(set) var aiLabel = "—"
    private(set) var lastInferenceMs: Double?
    /// Wall-clock time of `chooseAction` per move, as the app experiences it.
    private(set) var inferenceHistory: [Double] = []
    /// For `RemoteModelPlayer`: the server's own inference time per move, so
    /// the HTTP/JSON overhead is `inferenceHistory - serverInferenceHistory`.
    private(set) var serverInferenceHistory: [Double] = []
    /// Time to construct the player (model load + function lookup, or the
    /// `/info` + `/reset` round trip for the remote player).
    private(set) var loadMs: Double?
    private(set) var benchStatus: String?
    private(set) var scores: [AIKind: Scoreboard] = [:]
    var tickInterval: Duration = .milliseconds(140)

    /// Changing this restarts the game with the new opponent.
    var aiKind: AIKind = .minimax {
        didSet { if aiKind != oldValue, phase != .idle { start() } }
    }

    private var humanDirection: Direction?
    private var aiPlayer: (any SnakePlayer)?
    private var loop: Task<Void, Never>?
    private var seed: UInt64 = 1

    // MARK: - Controls

    func steer(_ d: Direction) { humanDirection = d }

    func start() {
        loop?.cancel()
        seed &+= 1
        game = SnakeGame(seed: seed)
        humanDirection = nil
        lastInferenceMs = nil
        inferenceHistory = []
        serverInferenceHistory = []
        loadMs = nil
        benchStatus = nil
        phase = .loading
        let kind = aiKind
        loop = Task { [weak self] in
            guard let self else { return }
            let t0 = ContinuousClock.now
            await self.loadPlayer(kind)
            self.loadMs = Self.ms(since: t0)
            guard !Task.isCancelled else { return }
            self.phase = .running
            await self.run(kind)
        }
    }

    func stop() {
        loop?.cancel()
        phase = .idle
    }

    // MARK: - Players

    private func loadPlayer(_ kind: AIKind) async {
        switch kind {
        case .heuristic:
            aiPlayer = HeuristicPlayer(epsilon: 0, seed: seed)
            aiLabel = "heuristic"
        case .minimax:
            aiPlayer = MinimaxPlayer(depth: 2)
            aiLabel = "minimax (depth 2)"
        case .model, .modelMinimaxTaught, .modelMixedTaught:
            let (player, label) = await loadModelPlayer(port: kind.serverPort!)
            aiPlayer = player
            aiLabel = label
        }
    }

    /// Prefer the in-process CoreAI.framework player (macOS 27+); otherwise the
    /// same asset served by `python -m snake_ai.serve`. If neither is
    /// available, play minimax and say so in the HUD.
    private func loadModelPlayer(port: Int) async -> (any SnakePlayer, String) {
        var reasons: [String] = []
        if port == 8765, let (url, function) = Self.locateModel() {
            do {
                let p = try await ModelPlayer(modelURL: url, functionName: function)
                return (p, "Core AI · \(url.lastPathComponent) (in-process)")
            } catch ModelError.coreAIUnavailable {
                reasons.append("CoreAI.framework needs macOS 27")
            } catch {
                reasons.append("in-process load failed: \(error)")
            }
        } else if port == 8765 {
            reasons.append("no .aimodel found")
        }
        do {
            let p = try await RemoteModelPlayer(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
            let tag = p.info.tag.isEmpty ? "" : " [\(p.info.tag)]"
            return (p, "Core AI · \(p.info.asset)\(tag) via Python runtime :\(port)")
        } catch {
            reasons.append("no model server on :\(port) — run: python -m snake_ai.serve --port \(port)")
        }
        return (MinimaxPlayer(depth: 2), "minimax fallback — " + reasons.joined(separator: "; "))
    }

    // MARK: - Loop

    private func run(_ kind: AIKind) async {
        while !Task.isCancelled && !game.isOver {
            var actions: [Int: Direction] = [:]
            if game.snakes[0].alive, var p = aiPlayer {
                let t0 = ContinuousClock.now
                do {
                    actions[0] = try await p.chooseAction(game: game, snakeID: 0)
                } catch {
                    aiLabel = "AI error: \(error)"
                }
                let ms = Self.ms(since: t0)
                lastInferenceMs = ms
                inferenceHistory.append(ms)
                if let remote = p as? RemoteModelPlayer {
                    serverInferenceHistory.append(remote.lastInferenceMs)
                }
                aiPlayer = p
            }
            if game.snakes[1].alive, let d = humanDirection { actions[1] = d }
            game.step(actions)
            try? await Task.sleep(for: tickInterval)
        }
        guard !Task.isCancelled else { return }
        var board = scores[kind, default: Scoreboard()]
        switch game.winner {
        case 0: board.aiWins += 1; phase = .over("AI wins")
        case 1: board.humanWins += 1; phase = .over("You win!")
        default: board.draws += 1; phase = .over("Draw")
        }
        scores[kind] = board
    }

    var averageInferenceMs: Double? {
        inferenceHistory.isEmpty ? nil : inferenceHistory.reduce(0, +) / Double(inferenceHistory.count)
    }

    private static func ms(since t0: ContinuousClock.Instant) -> Double {
        let c = (ContinuousClock.now - t0).components
        return Double(c.seconds) * 1e3 + Double(c.attoseconds) / 1e15
    }

    // MARK: - Bench export

    /// Write this game's load + inference numbers as a `snake-bench/1` record
    /// (see Bench.swift) next to the Python baseline in `docs/bench/`.
    func saveBench() {
        guard let aiPlayer, let loadMs, !inferenceHistory.isEmpty else {
            benchStatus = "nothing to save yet"
            return
        }
        let player = BenchPlayer(
            player: aiKind.rawValue, label: aiLabel, games: 1,
            wins: game.winner == 0 ? 1 : 0, draws: game.isOver && game.winner == nil ? 1 : 0,
            avgSteps: Double(game.stepCount), loads: [loadMs], inference: inferenceHistory,
            serverInference: aiPlayer is RemoteModelPlayer ? serverInferenceHistory : nil)
        let tickMs = tickInterval.components.seconds * 1000 + tickInterval.components.attoseconds / 1_000_000_000_000_000
        let record = BenchRecord(
            runtime: BenchRecord.runtimeDescription(for: aiPlayer) + " · SnakeApp, human opponent",
            config: ["tick_ms": .int(Int(tickMs)), "seed": .int(Int(seed)), "safe_only": true],
            players: [player])
        let slug = aiKind.rawValue.lowercased().replacing(/[^a-z0-9]+/, with: "-")
        let stamp = Date().formatted(.iso8601.year().month().day().dateSeparator(.dash))
        let url = BenchRecord.defaultDirectory().appending(path: "app-\(slug)-\(stamp).json")
        do {
            try record.write(to: url)
            benchStatus = "saved \(url.path)"
        } catch {
            benchStatus = "save failed: \(error)"
        }
    }

    // MARK: - Model discovery

    /// `SNAKE_MODEL` env var (function name from `SNAKE_MODEL_FUNCTION`, default
    /// `main`), else look upward from the working directory for the repo's
    /// static-shape decode asset, then the dynamic stateful one.
    static func locateModel() -> (URL, String)? {
        let env = ProcessInfo.processInfo.environment
        if let p = env["SNAKE_MODEL"] {
            return (URL(fileURLWithPath: p), env["SNAKE_MODEL_FUNCTION"] ?? "main")
        }
        let candidates = [("models/SnakeTransformerDecode.aimodel", "main_decode"),
                          ("models/SnakeTransformerStateful.aimodel", "main")]
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<4 {
            for (rel, fn) in candidates {
                let candidate = dir.appending(path: rel)
                if FileManager.default.fileExists(atPath: candidate.path) { return (candidate, fn) }
            }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }
}
