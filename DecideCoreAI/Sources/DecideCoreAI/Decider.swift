import Foundation

#if canImport(CoreAI)
import CoreAI

/// In-process decider on `CoreAI.framework`, written against the same API
/// surface as `SnakeCoreAI/ModelPlayer.swift` (`AIModel(contentsOf:)`,
/// `loadFunction(named:)`, `NDArray`, `InferenceFunction.run(inputs:)`).
/// Takes pre-tokenized ids for now; the Swift BPE port is step 2.
@available(macOS 27, iOS 27, *)
public struct Decider: Sendable {
    let function: InferenceFunction
    public let functionName: String
    public let temperature: Double

    /// `functionName` is `main` for the dynamic asset or `main_n{N}_l{L}` for
    /// the static one; read `models/decide/calibration.json` for `temperature`.
    public init(modelURL: URL, functionName: String = "main", temperature: Double = 1.0) async throws {
        try self.init(model: try await AIModel(contentsOf: modelURL), functionName: functionName, temperature: temperature)
    }

    /// From an already-loaded model, so one load of the static asset serves every `main_n{N}_l{L}`.
    public init(model: AIModel, functionName: String = "main", temperature: Double = 1.0) throws {
        guard let fn = try model.loadFunction(named: functionName) else {
            throw DeciderError.missingFunction(functionName)
        }
        self.function = fn
        self.functionName = functionName
        self.temperature = temperature
    }

    /// `inputIDs` / `attentionMask` are `[N][L]` int32, right-padded (pad id 1, mask 0).
    /// Returns `[N][3]` logits in `[contradiction, entailment, neutral]` order.
    public func logits(inputIDs: [[Int32]], attentionMask: [[Int32]]) async throws -> [[Float]] {
        let n = inputIDs.count, l = inputIDs.first?.count ?? 0
        let ids = NDArray(scalars: inputIDs.joined(), shape: [n, l])
        let mask = NDArray(scalars: attentionMask.joined(), shape: [n, l])
        var outputs = try await function.run(inputs: ["input_ids": ids, "attention_mask": mask])
        guard let out = outputs.remove("logits")?.ndArray else { throw DeciderError.missingOutput("logits") }
        // Read through the strides: the runtime may hand back a non-contiguous (e.g. padded) buffer.
        return out.view(as: Float.self).withUnsafePointer { p, _, strides in
            let rs = strides[0], cs = strides[1]
            return (0..<n).map { r in (0..<3).map { p[r * rs + $0 * cs] } }
        }
    }

    /// P(yes) = softmax(logits / T)[entailment], the same post-processing as `decide_ai.decider`.
    public func probabilities(inputIDs: [[Int32]], attentionMask: [[Int32]]) async throws -> [Double] {
        try await logits(inputIDs: inputIDs, attentionMask: attentionMask).map { row in
            let scaled = row.map { Double($0) / temperature }
            let m = scaled.max() ?? 0
            let e = scaled.map { exp($0 - m) }
            return e[1] / e.reduce(0, +)
        }
    }
}

#else

/// CoreAI.framework is not in this SDK (requires macOS 27 / Xcode 27). Keep
/// the public surface so the CLI compiles; it fails at runtime with a clear error.
public struct Decider {
    public let functionName: String
    public let temperature: Double
    public init(modelURL: URL, functionName: String = "main", temperature: Double = 1.0) async throws {
        throw DeciderError.coreAIUnavailable
    }
    public func logits(inputIDs: [[Int32]], attentionMask: [[Int32]]) async throws -> [[Float]] {
        throw DeciderError.coreAIUnavailable
    }
    public func probabilities(inputIDs: [[Int32]], attentionMask: [[Int32]]) async throws -> [Double] {
        throw DeciderError.coreAIUnavailable
    }
}

#endif
