import Foundation

/// One generated token plus the timing the gauges need.
public struct GeneratedToken: Sendable {
    public let id: Int
    public let text: String
    /// Wall-clock ms for producing this token, as measured by the generator
    /// (server-side for `RemoteGenerator`, in-process for `ModelGenerator`).
    public let ms: Double
}

/// A generator owns its KV-cache state for one conversation: `prefill` fills
/// the cache from the prompt, then `generate` yields one token per decode
/// step until EOS or `maxTokens`. Mirrors the two Core AI functions the
/// asset will expose (`prefill` with dynamic T, `decode` pinned to `[1, 1]`).
public protocol TokenGenerator: Sendable {
    /// Human-readable description of what is running and where.
    var label: String { get }
    /// Reset the caches so the next `generate` starts a fresh conversation.
    mutating func reset() async throws
    /// Stream tokens for `prompt`.
    func generate(prompt: String, maxTokens: Int) -> AsyncThrowingStream<GeneratedToken, Error>
}

public enum GeneratorError: Error, CustomStringConvertible {
    case missingFunction(String)
    case missingOutput(String)
    case contextExhausted
    case server(String)

    public var description: String {
        switch self {
        case .missingFunction(let n): return "asset has no function named \(n)"
        case .missingOutput(let n): return "function returned no output named \(n)"
        case .contextExhausted: return "KV cache is full; call reset()"
        case .server(let m): return "server: \(m)"
        }
    }
}
