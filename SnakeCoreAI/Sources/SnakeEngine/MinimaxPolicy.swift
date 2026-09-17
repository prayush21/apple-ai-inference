/// Adversarial lookahead over a territory evaluation. Port of `snake_ai/minimax.py`.
///
/// `HeuristicPolicy` scores one move ahead and treats the opponent as static;
/// it dies exclusively by being sealed into a pocket by an opponent it never
/// modelled. This policy searches (my move, opponent reply) pairs with
/// alpha-beta pruning and evaluates positions by Voronoi territory, a trapped
/// penalty, length difference and food proximity. Against the greedy heuristic
/// it wins ~90–100% of games depending on the step limit.
public struct EvalWeights: Sendable {
    public var win: Double = 10_000
    public var territory: Double = 1
    public var trapped: Double = 200
    public var length: Double = 30
    public var food: Double = 15
    public init() {}
}

public struct MinimaxPolicy: Sendable {
    public var depth: Int
    public var weights: EvalWeights
    public private(set) var nodes = 0

    /// `depth` counts my own moves: 1 = 2-ply (~1.5 ms/move in Python, far
    /// faster here), 2 = 4-ply.
    public init(depth: Int = 2, weights: EvalWeights = EvalWeights()) {
        self.depth = depth
        self.weights = weights
    }

    public mutating func choose(game: SnakeGame, snakeID: Int) -> Direction {
        nodes = 0
        var best = game.snakes[snakeID].direction
        var bestScore = -Double.infinity
        for d in Self.orderedMoves(game, snakeID) {
            let score = opponentReply(game, me: snakeID, myMove: d, depth: depth, alpha: bestScore, beta: .infinity)
            if score > bestScore { (best, bestScore) = (d, score) }
        }
        return best
    }

    // MARK: - Search

    /// Opponent minimises over its replies to `myMove`.
    private mutating func opponentReply(_ game: SnakeGame, me: Int, myMove: Direction, depth: Int, alpha: Double, beta: Double) -> Double {
        let opp = 1 - me
        var value = Double.infinity
        var beta = beta
        for od in Self.orderedMoves(game, opp) {
            var child = game  // value type: the copy is the clone
            child.step([me: myMove, opp: od])
            nodes += 1
            let v = (child.isOver || depth <= 1)
                ? evaluate(child, me: me)
                : myTurn(child, me: me, depth: depth - 1, alpha: alpha, beta: beta)
            value = min(value, v)
            beta = min(beta, value)
            if beta <= alpha { break }
        }
        return value
    }

    private mutating func myTurn(_ game: SnakeGame, me: Int, depth: Int, alpha: Double, beta: Double) -> Double {
        var value = -Double.infinity
        var alpha = alpha
        for d in Self.orderedMoves(game, me) {
            let v = opponentReply(game, me: me, myMove: d, depth: depth, alpha: alpha, beta: beta)
            value = max(value, v)
            alpha = max(alpha, value)
            if beta <= alpha { break }
        }
        return value
    }

    /// Safe moves first, continuing straight first (better pruning); if none
    /// are safe, the current direction so the snake still "moves".
    static func orderedMoves(_ game: SnakeGame, _ snakeID: Int) -> [Direction] {
        let snake = game.snakes[snakeID]
        guard snake.alive else { return [snake.direction] }
        var safe = Direction.allCases.filter { game.isSafe(snakeID, $0) }
        if safe.isEmpty { return [snake.direction] }
        safe.sort { a, b in (a == snake.direction) && !(b == snake.direction) }
        return safe
    }

    // MARK: - Evaluation

    /// Static score of `game` from snake `me`'s point of view.
    public func evaluate(_ game: SnakeGame, me: Int) -> Double {
        let w = weights
        let mine = game.snakes[me], opp = game.snakes[1 - me]
        if !mine.alive && !opp.alive { return 0 }
        if !mine.alive { return -w.win }
        if !opp.alive { return w.win }
        if game.stepCount >= game.maxSteps {  // timeout: longer snake wins
            return w.win * (mine.length > opp.length ? 1 : mine.length < opp.length ? -1 : 0)
        }

        let (myTerr, theirTerr) = Self.voronoi(game, me: me)
        var score = w.territory * Double(myTerr - theirTerr)

        // A region smaller than my body is a slow death even if no move is fatal yet.
        let reach = Self.bfs(game, from: mine.head, occupied: game.occupiedCells())
        if reach.count - 1 < mine.length { score -= w.trapped }

        score += w.length * Double(mine.length - opp.length)

        // Food: reward being closer than the opponent, and being close at all,
        // both normalised by board size so the term is in [-2, 2].
        let scale = Double(game.width + game.height)
        let myD = Double(abs(game.food.x - mine.head.x) + abs(game.food.y - mine.head.y)) / scale
        let theirD = Double(abs(game.food.x - opp.head.x) + abs(game.food.y - opp.head.y)) / scale
        score += w.food * ((theirD - myD) + (1 - myD))
        return score
    }

    /// (cells I reach strictly first, cells the opponent reaches strictly first).
    public static func voronoi(_ game: SnakeGame, me: Int) -> (Int, Int) {
        let occupied = game.occupiedCells()
        let distA = bfs(game, from: game.snakes[me].head, occupied: occupied)
        let distB = bfs(game, from: game.snakes[1 - me].head, occupied: occupied)
        var mine = 0, theirs = 0
        for (p, da) in distA where distB[p].map({ da < $0 }) ?? true { mine += 1 }
        for (p, db) in distB where distA[p].map({ db < $0 }) ?? true { theirs += 1 }
        return (mine, theirs)
    }

    static func bfs(_ game: SnakeGame, from start: Point, occupied: Set<Point>) -> [Point: Int] {
        var dist: [Point: Int] = [start: 0]
        var queue = [start]
        var i = 0
        while i < queue.count {
            let p = queue[i]; i += 1
            let d = dist[p]! + 1
            for dir in Direction.allCases {
                let n = Point(p.x + dir.delta.dx, p.y + dir.delta.dy)
                if game.inBounds(n), !occupied.contains(n), dist[n] == nil {
                    dist[n] = d
                    queue.append(n)
                }
            }
        }
        return dist
    }
}
