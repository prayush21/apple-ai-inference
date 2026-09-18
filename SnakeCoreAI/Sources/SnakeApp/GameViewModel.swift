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
    private(set) var inferenceHistory: [Double] = []
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
        phase = .loading
        let kind = aiKind
        loop = Task { [weak self] in
            guard let self else { return }
            await self.loadPlayer(kind)
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
                let ms = Double((ContinuousClock.now - t0).components.attoseconds) / 1e15
                lastInferenceMs = ms
                inferenceHistory.append(ms)
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
