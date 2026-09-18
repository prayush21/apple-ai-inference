import Foundation

/// Streams tokens from `python -m llm_ai.serve` over HTTP (server-sent
/// events), so the app can drive the model through the Core AI Python
/// runtime before macOS 27. The server owns the KV-cache states; this type
/// only owns the connection.
///
/// Stub for milestone 1: `init` probes `/info` so the CLI and app can already
/// tell whether a server is up. `generate` is wired at milestone 6.
public struct RemoteGenerator: TokenGenerator {
    public struct Info: Decodable, Sendable {
        public let asset: String
        public let tag: String
        public let maxSeqLen: Int
    }

    public let baseURL: URL
    public let info: Info
    public var label: String { "llm_ai.serve @ \(baseURL.host() ?? "?"):\(baseURL.port ?? 0) — \(info.asset)" }

    public init(baseURL: URL = URL(string: "http://127.0.0.1:8770")!) async throws {
        self.baseURL = baseURL
        var req = URLRequest(url: baseURL.appending(path: "info"))
        req.timeoutInterval = 2
        let (data, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw GeneratorError.server(String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.info = try decoder.decode(Info.self, from: data)
    }

    public mutating func reset() async throws {}

    public func generate(prompt: String, maxTokens: Int) -> AsyncThrowingStream<GeneratedToken, Error> {
        AsyncThrowingStream { $0.finish(throwing: GeneratorError.server("streaming not implemented yet")) }
    }
}
