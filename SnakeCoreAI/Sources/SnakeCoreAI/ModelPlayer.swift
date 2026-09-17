import Foundation
import SnakeEngine

#if canImport(CoreAI)
import CoreAI

// MARK: - Stateless player (full history every step)

/// Runs `SnakeTransformer.aimodel`: accumulates the whole game history and
/// feeds `features[1, T, 16]` on every move. Simple, but inference latency
/// grows with T (see the Core AI Instrument / debug gauge).
@available(macOS 27, iOS 27, *)
public struct StatelessModelPlayer: SnakePlayer {
    let nextActionFunction: InferenceFunction
    private var history: [[Float]] = []

    /// Initialize the player by loading the AIModel and InferenceFunction.
    /// Do this when preparing the feature (e.g. app launch), not per move.
    public init(modelURL: URL) async throws {
        let model = try await AIModel(contentsOf: modelURL)
        guard let function = try model.loadFunction(named: "main") else {
            throw ModelError.missingFunction("main")
        }
        self.nextActionFunction = function
    }

    public mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction {
        history.append(FeatureExtractor.features(of: game, for: snakeID))

        // Create an NDArray for the next input and write board features into it.
        var inputFeatures = NDArray(shape: [1, history.count, FeatureExtractor.featureDim], scalarType: .float32)
        inputFeatures.mutableView(as: Float.self).write(rows: history)

        // Run inference and extract the expected logits output NDArray.
        var outputs = try await nextActionFunction.run(inputs: ["features": inputFeatures])
        guard let logits = outputs.remove("logits")?.ndArray else {
            throw ModelError.missingOutput("logits")
        }
        return predictedDirection(fromLogits: logits.view(as: Float.self).lastRow(width: 4), game: game, snakeID: snakeID)
    }
}

// MARK: - Stateful player (KV-cache states)

/// Runs `SnakeTransformerStateful.aimodel` (function `main`, dynamic step
/// count) or `SnakeTransformerDecode.aimodel` (function `main_decode`, inputs
/// pinned to one step — skips per-call type inference). The key/value caches
/// are NDArrays owned by the player and passed as *states*: the runtime reads
/// and updates them in place, so each move only sends the newest board
/// features and its position. Latency stays flat for the whole game.
@available(macOS 27, iOS 27, *)
public struct ModelPlayer: SnakePlayer {
    let nextActionFunction: InferenceFunction

    var keyCache: NDArray
    var valueCache: NDArray
    private let maxContext: Int
    private var position = 0

    public init(modelURL: URL, functionName: String = "main") async throws {
        let model = try await AIModel(contentsOf: modelURL)
        guard let function = try model.loadFunction(named: functionName) else {
            throw ModelError.missingFunction(functionName)
        }
        self.nextActionFunction = function

        // The model was converted with fixed-size caches for a maximum context
        // length. Read the exact shape from the function's state descriptor
        // instead of hard-coding [layers, 1, maxContext, hiddenDim].
        let keyShape = function.stateDescriptor(named: "keyCache").shape
        let valueShape = function.stateDescriptor(named: "valueCache").shape
        self.keyCache = NDArray(shape: keyShape, scalarType: .float32)
        self.valueCache = NDArray(shape: valueShape, scalarType: .float32)
        self.maxContext = keyShape[2]
    }

    public mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction {
        guard position < maxContext else { throw ModelError.contextExhausted }

        // Only the newest board state is needed; history lives in the caches.
        var inputFeatures = NDArray(shape: [1, 1, FeatureExtractor.featureDim], scalarType: .float32)
        inputFeatures.mutableView(as: Float.self).write(rows: [FeatureExtractor.features(of: game, for: snakeID)])

        // position_ids is int32 in the converted model (coreai-torch maps int64 -> int32).
        var positionIDs = NDArray(shape: [1, 1], scalarType: .int32)
        positionIDs.mutableView(as: Int32.self).write(rows: [[Int32(position)]])

        // Views of the caches are handed to the runtime as mutable state.
        var stateViews = InferenceFunction.MutableViews()
        stateViews.insert(&keyCache, for: "keyCache")
        stateViews.insert(&valueCache, for: "valueCache")

        var outputs = try await nextActionFunction.run(
            inputs: ["features": inputFeatures, "position_ids": positionIDs],
            states: stateViews)
        guard let logits = outputs.remove("logits")?.ndArray else {
            throw ModelError.missingOutput("logits")
        }
        position += 1
        return predictedDirection(fromLogits: logits.view(as: Float.self).lastRow(width: 4), game: game, snakeID: snakeID)
    }
}

// MARK: - Specialization helpers (from the "Specialization" section of the talk)

@available(macOS 27, iOS 27, *)
public enum ModelPreparation {
    /// True if the model is already specialized for this device and cached, so
    /// loading it will be fast. Use this to gate the AI feature or to warn the
    /// user that preparation may take a while.
    public static func isReady(modelURL: URL) throws -> Bool {
        try AIModelCache.default.model(for: modelURL, options: .default) != nil
    }

    /// Explicitly specialize ahead of time (e.g. after download or opt-in),
    /// independent of loading the model into a player.
    public static func prepare(modelURL: URL) async throws {
        try await AIModel.specialize(contentsOf: modelURL)
    }
}

// MARK: - View helpers
//
// The session shows `NDArray.MutableView<Float>` / `NDArray.View<Float>` being
// passed around but not their element API. These two helpers are the only
// places that touch elements; adjust them against the Xcode 27 SDK if the
// accessor names differ (e.g. subscript vs. withUnsafeBufferPointer).

@available(macOS 27, iOS 27, *)
extension NDArray.MutableView {
    /// Write a row-major `[rows][cols]` table into a view of shape `[1, rows, cols]` or `[rows, cols]`.
    mutating func write(rows: [[Element]]) {
        var i = 0
        for row in rows {
            for v in row {
                self[i] = v
                i += 1
            }
        }
    }
}

@available(macOS 27, iOS 27, *)
extension NDArray.View {
    /// Last `width` elements of a row-major buffer (the logits for the newest step).
    func lastRow(width: Int) -> [Element] {
        let n = count
        return (n - width..<n).map { self[$0] }
    }
}

#else

// CoreAI.framework is not in this SDK (requires macOS 27 / Xcode 27). Keep the
// public surface so the CLI compiles; it fails at runtime with a clear error.

public struct ModelPlayer: SnakePlayer {
    public init(modelURL: URL, functionName: String = "main") async throws { throw ModelError.coreAIUnavailable }
    public mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction {
        throw ModelError.coreAIUnavailable
    }
}

public typealias StatelessModelPlayer = ModelPlayer

#endif
