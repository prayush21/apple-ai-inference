import Foundation
import Observation
import SnakeCoreAI
import SnakeEngine

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
    var tickInterval: Duration = .milliseconds(140)

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
        loop = Task { [weak self] in
            guard let self else { return }
            await self.loadPlayer()
            self.phase = .running
            await self.run()
        }
    }

    func stop() {
        loop?.cancel()
        phase = .idle
    }

    // MARK: - Loop

    private func loadPlayer() async {
        if aiPlayer == nil {
            if let (url, function) = Self.locateModel() {
                do {
                    aiPlayer = try await ModelPlayer(modelURL: url, functionName: function)
                    aiLabel = "Core AI · \(url.lastPathComponent)"
                } catch ModelError.coreAIUnavailable {
                    aiLabel = "heuristic (CoreAI.framework needs macOS 27)"
                } catch {
                    aiLabel = "heuristic (model failed: \(error))"
                }
            } else {
                aiLabel = "heuristic (no .aimodel found; set SNAKE_MODEL)"
            }
            if aiPlayer == nil { aiPlayer = HeuristicPlayer(epsilon: 0, seed: seed) }
        } else if aiPlayer is HeuristicPlayer {
            aiPlayer = HeuristicPlayer(epsilon: 0, seed: seed)
        } else if let (url, function) = Self.locateModel(),
                  let fresh = try? await ModelPlayer(modelURL: url, functionName: function) {
            // A stateful model player owns KV caches: start each game with fresh ones.
            aiPlayer = fresh
        }
    }

    private func run() async {
        while !Task.isCancelled && !game.isOver {
            var actions: [Int: Direction] = [:]
            if game.snakes[0].alive, var p = aiPlayer {
                let t0 = ContinuousClock.now
                if let d = try? await p.chooseAction(game: game, snakeID: 0) { actions[0] = d }
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
        switch game.winner {
        case 0: phase = .over("AI wins")
        case 1: phase = .over("You win!")
        default: phase = .over("Draw")
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
