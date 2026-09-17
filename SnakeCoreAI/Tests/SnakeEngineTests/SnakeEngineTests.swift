import Foundation
import Testing
@testable import SnakeEngine

// MARK: - Rules (mirrors tests/test_game.py)

@Test func initialLayout() {
    let g = SnakeGame(width: 12, height: 12, seed: 0)
    #expect(g.snakes[0].head == Point(2, 6) && g.snakes[0].direction == .right)
    #expect(g.snakes[1].head == Point(9, 6) && g.snakes[1].direction == .left)
    #expect(!g.occupiedCells().contains(g.food))
}

@Test func wallCollisionEndsGame() {
    var g = SnakeGame(width: 6, height: 6, seed: 0)
    for _ in 0..<10 { g.step([0: .up, 1: .down]) }
    #expect(!g.snakes[1].alive && g.snakes[0].alive)
    #expect(g.isOver && g.winner == 0)
}

@Test func reverseIsIgnored() {
    var g = SnakeGame(seed: 0)
    g.step([0: .left])
    #expect(g.snakes[0].direction == .right)
    #expect(g.snakes[0].head == Point(3, 6))
}

@Test func eatingGrows() {
    var g = SnakeGame(width: 12, height: 12, snakes: SnakeGame(seed: 0).snakes, food: Point(3, 6))
    g.step([:])
    #expect(g.snakes[0].length == 4)
    #expect(g.food != Point(3, 6))
}

@Test func heuristicSurvivesAWhile() {
    var g = SnakeGame(seed: 3, maxSteps: 100)
    var p = HeuristicPolicy()
    while !g.isOver { g.step([0: p.choose(game: g, snakeID: 0), 1: p.choose(game: g, snakeID: 1)]) }
    #expect(g.stepCount > 10)
}

// MARK: - Feature parity with the Python engine

private struct FixtureSnake: Decodable {
    let body: [[Int]]
    let direction: Int
    let alive: Bool
}

private struct FixtureCase: Decodable {
    let width: Int
    let height: Int
    let food: [Int]
    let snakes: [FixtureSnake]
    let features: [[Float]?]
}

@Test func featuresMatchPythonBitForBit() throws {
    let url = try #require(Bundle.module.url(forResource: "features", withExtension: "json", subdirectory: "Fixtures"))
    let cases = try JSONDecoder().decode([FixtureCase].self, from: Data(contentsOf: url))
    #expect(cases.count > 0)

    for c in cases {
        let snakes = c.snakes.map {
            Snake(body: $0.body.map { Point($0[0], $0[1]) }, direction: Direction(rawValue: $0.direction)!, alive: $0.alive)
        }
        let game = SnakeGame(width: c.width, height: c.height, snakes: snakes, food: Point(c.food[0], c.food[1]))
        for id in 0..<2 {
            guard let expected = c.features[id] else { continue }
            let got = FeatureExtractor.features(of: game, for: id)
            #expect(got == expected, "snake \(id) features differ: \(got) vs \(expected)")
        }
    }
}

// MARK: - Minimax teacher

@Test func minimaxBeatsHeuristic() {
    var wins = 0
    for seed in 0..<12 {
        var g = SnakeGame(seed: UInt64(seed), maxSteps: 300)
        var a = MinimaxPolicy(depth: 1)
        var b = HeuristicPolicy(epsilon: 0.05, seed: UInt64(seed))
        while !g.isOver {
            let d = a.choose(game: g, snakeID: 0)
            #expect(g.isSafe(0, d) || !Direction.allCases.contains { g.isSafe(0, $0) })
            g.step([0: d, 1: b.choose(game: g, snakeID: 1)])
        }
        if g.winner == 0 { wins += 1 }
    }
    #expect(wins >= 9, "minimax won only \(wins)/12")
}

@Test func voronoiIsSymmetricAtStart() {
    let g = SnakeGame(seed: 0)
    let (mine, theirs) = MinimaxPolicy.voronoi(g, me: 0)
    #expect(mine == theirs)
}
