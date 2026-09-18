import Foundation
import SnakeCoreAI
import SnakeEngine

// snake-cli [--model path.aimodel] [--ai heuristic|minimax|model] [--games N] [--render] [--seed S] [--json out.json]
//
// --ai model uses the Core AI model served by `python -m snake_ai.serve`.
// --json writes load + per-move inference stats in the same `snake-bench/1`
//   schema as `python -m snake_ai.play --json`, for docs/bench/.
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
var jsonPath: String?
while let a = args.popFirst() {
    switch a {
    case "--model": modelPath = args.popFirst()
    case "--ai": ai = args.popFirst() ?? "minimax"
    case "--games": games = Int(args.popFirst() ?? "1") ?? 1
    case "--render": render = true
    case "--seed": seed = UInt64(args.popFirst() ?? "100") ?? 100
    case "--json": jsonPath = args.popFirst()
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

func ms(since t0: ContinuousClock.Instant) -> Double {
    let c = (ContinuousClock.now - t0).components
    return Double(c.seconds) * 1e3 + Double(c.attoseconds) / 1e15
}

var wins = 0, draws = 0, totalSteps = 0
var loads: [Double] = [], inference: [Double] = [], serverInference: [Double] = []
var runtime = "no model"
for g in 0..<games {
    let gameSeed = seed + UInt64(g)
    var game = SnakeGame(seed: gameSeed)
    let t0 = ContinuousClock.now
    var player0 = try await makePlayer0(modelPath: modelPath, ai: ai, seed: gameSeed)
    loads.append(ms(since: t0))
    runtime = BenchRecord.runtimeDescription(for: player0)
    var player1 = HeuristicPlayer(epsilon: 0.05, seed: gameSeed)

    while !game.isOver {
        var actions: [Int: Direction] = [:]
        if game.snakes[0].alive {
            let t0 = ContinuousClock.now
            actions[0] = try await player0.chooseAction(game: game, snakeID: 0)
            inference.append(ms(since: t0))
            if let remote = player0 as? RemoteModelPlayer { serverInference.append(remote.lastInferenceMs) }
        }
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
let avgSteps = Double(totalSteps) / Double(games)
let inf = BenchStats(inference), load = BenchLoad(loads)
print("games=\(games) snake0 wins=\(wins) draws=\(draws) avg steps=\(avgSteps)")
print(String(format: "load ms: first %.1f, rest %@ | inference ms: mean %.2f, p50 %.2f, p95 %.2f, first-5 %.2f, last-5 %@",
             load.first, load.restMean.map { String(format: "%.1f", $0) } ?? "n/a",
             inf.mean ?? 0, inf.p50 ?? 0, inf.p95 ?? 0, inf.first5 ?? 0,
             inf.last5.map { String(format: "%.2f", $0) } ?? "n/a"))
if let jsonPath {
    let player = BenchPlayer(player: modelPath != nil ? "model" : ai, label: runtime, games: games, wins: wins,
                             draws: draws, avgSteps: avgSteps, loads: loads, inference: inference,
                             serverInference: serverInference.isEmpty ? nil : serverInference)
    let record = BenchRecord(runtime: runtime + " · snake-cli, heuristic opponent",
                             config: ["games": .int(games), "seed": .int(Int(seed)), "safe_only": true],
                             players: [player])
    try record.write(to: URL(fileURLWithPath: jsonPath))
    print("wrote \(jsonPath)")
}
