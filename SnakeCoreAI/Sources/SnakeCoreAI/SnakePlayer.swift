import SnakeEngine

/// Anything that can pick the next move for a snake.
public protocol SnakePlayer: Sendable {
    mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction
}

/// The hand-written policy wrapped as a player (opponent / fallback).
public struct HeuristicPlayer: SnakePlayer {
    private var policy: HeuristicPolicy

    public init(epsilon: Double = 0.05, seed: UInt64 = 0) {
        policy = HeuristicPolicy(epsilon: epsilon, seed: seed)
    }

    public mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction {
        policy.choose(game: game, snakeID: snakeID)
    }
}

/// Adversarial-search player (the "stronger teacher"). Much harder to beat
/// than `HeuristicPlayer`; also the fallback for snake A when CoreAI is
/// unavailable.
public struct MinimaxPlayer: SnakePlayer {
    private var policy: MinimaxPolicy

    public init(depth: Int = 2) {
        policy = MinimaxPolicy(depth: depth)
    }

    public mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction {
        policy.choose(game: game, snakeID: snakeID)
    }
}

public enum ModelError: Error {
    case missingFunction(String)
    case missingOutput(String)
    case missingState(String)
    case contextExhausted
    case coreAIUnavailable
    case server(String)
}

/// Pick the action from 4 logits. With `safeOnly`, the best non-fatal move is
/// taken instead of a fatal argmax (matches `predicted_direction` in play.py).
public func predictedDirection(fromLogits logits: [Float], game: SnakeGame, snakeID: Int, safeOnly: Bool = true) -> Direction {
    let ranked = logits.indices.sorted { logits[$0] > logits[$1] }.map { Direction(rawValue: $0)! }
    if safeOnly, let d = ranked.first(where: { game.isSafe(snakeID, $0) }) { return d }
    return ranked[0]
}
