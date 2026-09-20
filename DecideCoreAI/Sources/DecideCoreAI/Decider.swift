import Foundation

#if canImport(CoreAI)
import CoreAI

/// In-process decider on `CoreAI.framework`, written against the same API
/// surface as `SnakeCoreAI/ModelPlayer.swift` (`AIModel(contentsOf:)`,
/// `loadFunction(named:)`, `NDArray`, `InferenceFunction.run(inputs:)`).
/// Takes pre-tokenized ids for now; the Swift BPE port is step 2.
@available(macOS 27, iOS 27, *)
public struct Decider {
    let function: InferenceFunction
    public let functionName: String
    public let temperature: Double

    /// `functionName` is `main` for the dynamic asset or `main_n{N}_l{L}` for
    /// the static one; read `models/decide/calibration.json` for `temperature`.
    public init(modelURL: URL, functionName: String = "main", temperature: Double = 1.0) async throws {
        let model = try await AIModel(contentsOf: modelURL)
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
        var ids = NDArray(shape: [n, l], scalarType: .int32)
        ids.mutableView(as: Int32.self).write(rows: inputIDs)
        var mask = NDArray(shape: [n, l], scalarType: .int32)
        mask.mutableView(as: Int32.self).write(rows: attentionMask)
        var outputs = try await function.run(inputs: ["input_ids": ids, "attention_mask": mask])
        guard let out = outputs.remove("logits")?.ndArray else { throw DeciderError.missingOutput("logits") }
        let flat = out.view(as: Float.self)
        return (0..<n).map { r in (0..<3).map { flat[r * 3 + $0] } }
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

@available(macOS 27, iOS 27, *)
extension NDArray.MutableView {
    mutating func write(rows: [[Element]]) {
        var i = 0
        for row in rows { for v in row { self[i] = v; i += 1 } }
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
