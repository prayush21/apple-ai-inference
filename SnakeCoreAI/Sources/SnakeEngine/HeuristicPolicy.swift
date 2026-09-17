/// Greedy food-seeking policy with a flood-fill safety check.
/// Port of `snake_ai/policy.py` (the opponent / training-data generator).
public struct HeuristicPolicy: Sendable {
    public var epsilon: Double
    private var rng: SplitMix64

    public init(epsilon: Double = 0, seed: UInt64 = 0) {
        self.epsilon = epsilon
        self.rng = SplitMix64(seed: seed)
    }

    public mutating func choose(game: SnakeGame, snakeID: Int) -> Direction {
        let snake = game.snakes[snakeID]
        let safe = Direction.allCases.filter { game.isSafe(snakeID, $0) }
        guard !safe.isEmpty else { return snake.direction }  // doomed either way

        if epsilon > 0, Double(rng.next() % 1_000_000) / 1_000_000 < epsilon {
            return safe[Int(rng.next() % UInt64(safe.count))]
        }

        let occupied = game.occupiedCells()
        var best = safe[0]
        var bestScore = -Double.infinity
        for d in safe {
            let n = game.nextHead(of: snakeID, moving: d)
            let area = floodFillArea(game, from: n, occupied: occupied, cap: snake.length * 2 + 8)
            let dist = abs(game.food.x - n.x) + abs(game.food.y - n.y)
            var score = Double(area) - 0.5 * Double(dist)
            if d == snake.direction { score += 0.01 }  // prefer going straight
            if score > bestScore { (best, bestScore) = (d, score) }
        }
        return best
    }
}

/// Number of free cells reachable from `start` (up to `cap`).
func floodFillArea(_ game: SnakeGame, from start: Point, occupied: Set<Point>, cap: Int) -> Int {
    guard game.inBounds(start), !occupied.contains(start) else { return 0 }
    var seen: Set<Point> = [start]
    var queue = [start]
    var i = 0
    while i < queue.count && seen.count < cap {
        let p = queue[i]; i += 1
        for d in Direction.allCases {
            let n = Point(p.x + d.delta.dx, p.y + d.delta.dy)
            if game.inBounds(n), !occupied.contains(n), !seen.contains(n) {
                seen.insert(n)
                queue.append(n)
            }
        }
    }
    return seen.count
}
