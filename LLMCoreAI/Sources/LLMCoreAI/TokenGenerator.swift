import Foundation

/// One generated token plus the timing the gauges need.
public struct GeneratedToken: Sendable {
    public let id: Int
    public let text: String
    /// Wall-clock ms for producing this token, as measured by the generator
    /// (server-side for `RemoteGenerator`, in-process for `ModelGenerator`).
    /// For the first token of a reply this is the prefill (time to first token).
    public let ms: Double
}

/// A generator owns its KV-cache state for one conversation: each `generate`
/// appends the prompt to the cache (prefill) and then yields one token per
/// decode step until EOS or `maxTokens`. `reset` starts a new conversation.
/// Mirrors the asset's two Core AI functions (`main_prefill_t64`, `main_decode`).
public protocol TokenGenerator: Sendable {
    /// Human-readable description of what is running and where.
    var label: String { get }
    /// Reset the caches so the next `generate` starts a fresh conversation.
    func reset() async throws
    /// Stream tokens for `prompt`.
    func generate(prompt: String, maxTokens: Int) -> AsyncThrowingStream<GeneratedToken, Error>
}

public enum GeneratorError: Error, CustomStringConvertible {
    case missingFunction(String)
    case missingOutput(String)
    case missingState(String)
    case contextExhausted
    case untokenizedPrompt(String)
    case server(String)

    public var description: String {
        switch self {
        case .missingFunction(let n): return "asset has no function named \(n)"
        case .missingOutput(let n): return "function returned no output named \(n)"
        case .missingState(let n): return "function has no state named \(n)"
        case .contextExhausted: return "KV cache is full; call reset()"
        case .untokenizedPrompt(let p):
            return "no Swift BPE encoder yet and \"\(p.prefix(40))\" is not in prompt_ids.json (python -m llm_ai.prompt_ids)"
        case .server(let m): return "server: \(m)"
        }
    }
}
