import Foundation

#if canImport(CoreAI)
import CoreAI

/// In-process generator on `CoreAI.framework`. Owns `keyCache` / `valueCache`
/// NDArrays and passes them as states to the asset's `prefill` and `decode`
/// functions, exactly like `SnakeCoreAI.ModelPlayer` — just with a tokenizer
/// in front and a vocabulary-sized argmax behind.
///
/// Written against the WWDC26 session-324 API; untestable until this machine
/// runs macOS 27. Filled in at milestone 5 once the asset exists.
@available(macOS 27, iOS 27, *)
public struct ModelGenerator: TokenGenerator {
    public let label = "CoreAI.framework (in-process)"

    public init(modelURL: URL) async throws {
        _ = modelURL
        fatalError("ModelGenerator: not implemented until the .aimodel exists (llm_ai.convert)")
    }

    public mutating func reset() async throws {}

    public func generate(prompt: String, maxTokens: Int) -> AsyncThrowingStream<GeneratedToken, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}
#endif

/// True when this binary was compiled against an SDK that ships CoreAI.framework.
public let hasCoreAIFramework: Bool = {
    #if canImport(CoreAI)
    return true
    #else
    return false
    #endif
}()
