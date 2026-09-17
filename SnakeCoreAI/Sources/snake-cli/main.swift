import Foundation
import SnakeCoreAI
import SnakeEngine

// snake-cli [--model path.aimodel] [--ai heuristic|minimax|model] [--games N] [--render] [--seed S]
//
// --ai model uses the Core AI model served by `python -m snake_ai.serve`.
//
// Snake 0 is the model (Core AI) when --model is given and the framework is
// available, otherwise the policy chosen by --ai (default minimax). Snake 1
// is always the greedy heuristic.

var args = CommandLine.arguments.dropFirst()
var modelPath: String?
var games = 1
var render = false
var seed: UInt64 = 100
var ai = "minimax"
while let a = args.popFirst() {
    switch a {
    case "--model": modelPath = args.popFirst()
    case "--ai": ai = args.popFirst() ?? "minimax"
    case "--games": games = Int(args.popFirst() ?? "1") ?? 1
    case "--render": render = true
    case "--seed": seed = UInt64(args.popFirst() ?? "100") ?? 100
    default: FileHandle.standardError.write("unknown argument \(a)\n".data(using: .utf8)!)
    }
}

func fallback(_ ai: String, seed: UInt64) async throws -> any SnakePlayer {
    switch ai {
    case "heuristic": return HeuristicPlayer(epsilon: 0, seed: seed)
    case "model":
        let p = try await RemoteModelPlayer()
        if seed == 100 { print("remote model: \(p.info.asset) / \(p.info.function) via Python runtime") }
        return p
    default: return MinimaxPlayer(depth: 2)
    }
}

func makePlayer0(modelPath: String?, ai: String, seed: UInt64) async throws -> any SnakePlayer {
    guard let modelPath else { return try await fallback(ai, seed: seed) }
    do {
        return try await ModelPlayer(modelURL: URL(fileURLWithPath: modelPath))
    } catch ModelError.coreAIUnavailable {
        print("CoreAI.framework unavailable in this SDK/OS (needs macOS 27); falling back to \(ai) for snake 0.")
        return try await fallback(ai, seed: seed)
    }
}

var wins = 0, draws = 0, totalSteps = 0
for g in 0..<games {
    let gameSeed = seed + UInt64(g)
    var game = SnakeGame(seed: gameSeed)
    var player0 = try await makePlayer0(modelPath: modelPath, ai: ai, seed: gameSeed)
    var player1 = HeuristicPlayer(epsilon: 0.05, seed: gameSeed)

    while !game.isOver {
        var actions: [Int: Direction] = [:]
        if game.snakes[0].alive { actions[0] = try await player0.chooseAction(game: game, snakeID: 0) }
        if game.snakes[1].alive { actions[1] = try await player1.chooseAction(game: game, snakeID: 1) }
        game.step(actions)
        if render {
            print("\u{1B}[2J\u{1B}[H", terminator: "")
            print("step \(game.stepCount)  A=\(game.snakes[0].length)  B=\(game.snakes[1].length)")
            print(game.render())
            try await Task.sleep(for: .milliseconds(80))
        }
    }
    totalSteps += game.stepCount
    switch game.winner {
    case 0: wins += 1
    case nil: draws += 1
    default: break
    }
}
print("games=\(games) snake0 wins=\(wins) draws=\(draws) avg steps=\(Double(totalSteps) / Double(games))")
