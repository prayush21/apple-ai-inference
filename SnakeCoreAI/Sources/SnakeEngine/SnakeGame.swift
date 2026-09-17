/// Two-player snake game engine.
///
/// Traditional snake rules: snakes grow by eating food and die when they hit a
/// wall, themselves, or the other snake. The last snake standing wins.
///
/// This is a line-for-line port of `snake_ai/game.py`. Keep the two in sync:
/// the model is trained on features computed by the Python engine, so any
/// divergence in rules or feature layout silently degrades the in-app model.

public struct Point: Hashable, Sendable, Codable {
    public var x: Int
    public var y: Int
    public init(_ x: Int, _ y: Int) { self.x = x; self.y = y }
}

/// Absolute movement direction. The raw value is the action index the model
/// predicts and the index used for one-hot encoding in features.
public enum Direction: Int, CaseIterable, Sendable, Codable {
    case up = 0, down = 1, left = 2, right = 3

    /// (dx, dy) with y growing downward.
    public var delta: (dx: Int, dy: Int) {
        switch self {
        case .up: (0, -1)
        case .down: (0, 1)
        case .left: (-1, 0)
        case .right: (1, 0)
        }
    }

    public var opposite: Direction {
        switch self {
        case .up: .down
        case .down: .up
        case .left: .right
        case .right: .left
        }
    }

    public var oneHotEncoding: [Float] {
        var v: [Float] = [0, 0, 0, 0]
        v[rawValue] = 1
        return v
    }
}

/// A snake is a list of body cells, head first.
public struct Snake: Sendable, Codable {
    public var body: [Point]
    public var direction: Direction
    public var alive: Bool = true

    public var head: Point { body[0] }
    public var length: Int { body.count }
}

/// Deterministic (given `seed`) two-snake game on a `width` x `height` grid.
///
/// Snake 0 starts on the left moving right; snake 1 starts on the right moving
/// left. Both snakes move simultaneously each `step`.
public struct SnakeGame: Sendable {
    public let width: Int
    public let height: Int
    public let initialLength: Int
    public let maxSteps: Int

    public private(set) var snakes: [Snake] = []
    public private(set) var food = Point(0, 0)
    public private(set) var stepCount = 0
    private var rng: SplitMix64

    public init(width: Int = 12, height: Int = 12, seed: UInt64 = 0, initialLength: Int = 3, maxSteps: Int = 256) {
        self.width = width
        self.height = height
        self.initialLength = initialLength
        self.maxSteps = maxSteps
        self.rng = SplitMix64(seed: seed)
        reset()
    }

    /// Construct an arbitrary board state (used by tests and fixtures).
    public init(width: Int, height: Int, snakes: [Snake], food: Point, stepCount: Int = 0, maxSteps: Int = 256) {
        self.width = width
        self.height = height
        self.initialLength = snakes.first?.length ?? 3
        self.maxSteps = maxSteps
        self.rng = SplitMix64(seed: 0)
        self.snakes = snakes
        self.food = food
        self.stepCount = stepCount
    }

    // MARK: - Setup

    public mutating func reset() {
        let mid = height / 2
        let left = (0..<initialLength).map { Point(initialLength - 1 - $0, mid) }
        let right = (0..<initialLength).map { Point(width - initialLength + $0, mid) }
        snakes = [
            Snake(body: left, direction: .right),
            Snake(body: right, direction: .left),
        ]
        stepCount = 0
        food = spawnFood()
    }

    private mutating func spawnFood() -> Point {
        let occupied = Set(snakes.flatMap(\.body))
        var free: [Point] = []
        for y in 0..<height {
            for x in 0..<width where !occupied.contains(Point(x, y)) {
                free.append(Point(x, y))
            }
        }
        guard !free.isEmpty else { return Point(-1, -1) }
        return free[Int(rng.next() % UInt64(free.count))]
    }

    // MARK: - Queries

    public func inBounds(_ p: Point) -> Bool {
        p.x >= 0 && p.x < width && p.y >= 0 && p.y < height
    }

    public func occupiedCells() -> Set<Point> {
        Set(snakes.filter(\.alive).flatMap(\.body))
    }

    public func nextHead(of snakeID: Int, moving direction: Direction) -> Point {
        let h = snakes[snakeID].head
        let d = direction.delta
        return Point(h.x + d.dx, h.y + d.dy)
    }

    /// Would moving `snakeID` in `direction` be immediately fatal?
    /// Tail cells count as occupied (conservative), matching the Python engine.
    public func isSafe(_ snakeID: Int, _ direction: Direction) -> Bool {
        let snake = snakes[snakeID]
        if direction == snake.direction.opposite && snake.length > 1 { return false }
        let p = nextHead(of: snakeID, moving: direction)
        return inBounds(p) && !occupiedCells().contains(p)
    }

    public var aliveIDs: [Int] { snakes.indices.filter { snakes[$0].alive } }

    public var isOver: Bool { aliveIDs.count <= 1 || stepCount >= maxSteps }

    /// Index of the surviving snake, or nil for a draw / not over.
    public var winner: Int? {
        let alive = aliveIDs
        if alive.count == 1 { return alive[0] }
        if alive.count == 2 && stepCount >= maxSteps {
            let (l0, l1) = (snakes[0].length, snakes[1].length)
            if l0 != l1 { return l0 > l1 ? 0 : 1 }
        }
        return nil
    }

    // MARK: - Step

    /// Advance the game one tick. Snakes with no action keep their direction;
    /// reversing into yourself is ignored.
    public mutating func step(_ actions: [Int: Direction]) {
        guard !isOver else { return }

        // 1. Resolve directions.
        for i in snakes.indices where snakes[i].alive {
            var d = actions[i] ?? snakes[i].direction
            if d == snakes[i].direction.opposite && snakes[i].length > 1 {
                d = snakes[i].direction
            }
            snakes[i].direction = d
        }

        // 2. New heads and whether each snake eats.
        var newHeads: [Int: Point] = [:]
        var eats: [Int: Bool] = [:]
        for i in aliveIDs {
            let h = nextHead(of: i, moving: snakes[i].direction)
            newHeads[i] = h
            eats[i] = h == food
        }

        // 3. Move bodies (grow if eating).
        for i in aliveIDs {
            snakes[i].body.insert(newHeads[i]!, at: 0)
            if eats[i] != true { snakes[i].body.removeLast() }
        }

        // 4. Simultaneous collision detection.
        var dead = Set<Int>()
        for i in aliveIDs {
            let h = newHeads[i]!
            if !inBounds(h) { dead.insert(i); continue }
            for j in aliveIDs {
                let cells = j == i ? snakes[j].body.dropFirst() : snakes[j].body[...]
                if cells.contains(h) { dead.insert(i); break }
            }
        }
        for i in dead { snakes[i].alive = false }

        // 5. Respawn food if eaten.
        if eats.values.contains(true) { food = spawnFood() }

        stepCount += 1
    }

    // MARK: - Display

    /// ASCII board: `A`/`a` snake 0 head/body, `B`/`b` snake 1, `*` food.
    public func render() -> String {
        var grid = Array(repeating: Array(repeating: Character("."), count: width), count: height)
        if inBounds(food) { grid[food.y][food.x] = "*" }
        for (i, s) in snakes.enumerated() {
            var (headCh, bodyCh): (Character, Character) = i == 0 ? ("A", "a") : ("B", "b")
            if !s.alive { (headCh, bodyCh) = ("x", "x") }
            for (k, p) in s.body.enumerated() where inBounds(p) {
                grid[p.y][p.x] = k == 0 ? headCh : bodyCh
            }
        }
        let border = "+" + String(repeating: "-", count: width) + "+"
        let rows = grid.map { "|" + String($0) + "|" }
        return ([border] + rows + [border]).joined(separator: "\n")
    }
}

/// Small deterministic PRNG so games are reproducible from a seed.
struct SplitMix64: Sendable {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
