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
        let inputFeatures = NDArray(scalars: history.joined(), shape: [1, history.count, FeatureExtractor.featureDim])

        // Run inference and extract the expected logits output NDArray.
        var outputs = try await nextActionFunction.run(inputs: ["features": inputFeatures])
        guard let logits = outputs.remove("logits")?.ndArray else {
            throw ModelError.missingOutput("logits")
        }
        return predictedDirection(fromLogits: lastStepLogits(logits), game: game, snakeID: snakeID)
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

    // Optional so a move can take them out (see `chooseAction`); nil only during a call.
    var keyCache: NDArray?
    var valueCache: NDArray?
    private let maxContext: Int
    private var position = 0

    public init(modelURL: URL, functionName: String = "main") async throws {
        let model = try await AIModel(contentsOf: modelURL)
        guard let function = try model.loadFunction(named: functionName) else {
            throw ModelError.missingFunction(functionName)
        }
        self.nextActionFunction = function

        // The model was converted with fixed-size caches for a maximum context
        // length. Allocate them from the function's state descriptors instead
        // of hard-coding [layers, 1, maxContext, hiddenDim].
        func stateArray(_ name: String) throws -> NDArray {
            guard case .ndArray(let d)? = function.descriptor.stateDescriptor(of: name) else {
                throw ModelError.missingState(name)
            }
            return NDArray(descriptor: d)
        }
        let keys = try stateArray("keyCache")
        self.keyCache = keys
        self.valueCache = try stateArray("valueCache")
        self.maxContext = keys.shape[2]
    }

    public mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction {
        guard position < maxContext else { throw ModelError.contextExhausted }

        // Only the newest board state is needed; history lives in the caches.
        let inputFeatures = NDArray(scalars: FeatureExtractor.features(of: game, for: snakeID), shape: [1, 1, FeatureExtractor.featureDim])

        // position_ids is int32 in the converted model (coreai-torch maps int64 -> int32).
        let positionIDs = NDArray(scalars: [Int32(position)], shape: [1, 1])

        // Views of the caches are handed to the runtime as mutable state. The
        // views borrow both caches at once, which exclusivity forbids on two
        // stored properties of `self`, so they move into locals for the call.
        // Moved, not copied: `var keys = keyCache` would leave two references
        // to the storage and taking the mutable view would copy both caches
        // every move (gotcha 19).
        guard var keys = keyCache.take(), var values = valueCache.take() else {
            throw ModelError.missingState("keyCache/valueCache")
        }
        defer {
            keyCache = keys
            valueCache = values
        }
        var stateViews = InferenceFunction.MutableViews()
        stateViews.insert(&keys, for: "keyCache")
        stateViews.insert(&values, for: "valueCache")

        var outputs = try await nextActionFunction.run(
            inputs: ["features": inputFeatures, "position_ids": positionIDs],
            states: stateViews)
        guard let logits = outputs.remove("logits")?.ndArray else {
            throw ModelError.missingOutput("logits")
        }
        position += 1
        return predictedDirection(fromLogits: lastStepLogits(logits), game: game, snakeID: snakeID)
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

// MARK: - Output helper

/// The 4 action logits of the newest step from `logits[1, T, 4]`. Reads
/// through the strides: the runtime may hand back a non-contiguous buffer.
@available(macOS 27, iOS 27, *)
func lastStepLogits(_ logits: NDArray) -> [Float] {
    logits.view(as: Float.self).withUnsafePointer { p, shape, strides in
        let base = (shape[1] - 1) * strides[1]
        return (0..<shape[2]).map { p[base + $0 * strides[2]] }
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
