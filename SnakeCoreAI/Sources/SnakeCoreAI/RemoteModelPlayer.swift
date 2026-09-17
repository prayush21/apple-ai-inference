import Foundation
import SnakeEngine

/// Runs the Core AI model through `python -m snake_ai.serve` (the Core AI
/// Python runtime) over local HTTP. Used until `CoreAI.framework` is available
/// in-process. The split mirrors `ModelPlayer` exactly: this side extracts the
/// 16 features and picks the safe argmax; the server only returns logits.
public struct RemoteModelPlayer: SnakePlayer {
    public struct Info: Decodable, Sendable {
        public let asset: String
        public let function: String
        public let variant: String
        public let tag: String
        public let states: [String]
    }

    private struct ActResponse: Decodable { let logits: [Float]; let ms: Double }

    public let baseURL: URL
    public let info: Info
    /// Server-side inference time of the last move (excludes the HTTP hop).
    public private(set) var lastInferenceMs: Double = 0

    public init(baseURL: URL = URL(string: "http://127.0.0.1:8765")!) async throws {
        self.baseURL = baseURL
        var req = URLRequest(url: baseURL.appending(path: "info"))
        req.timeoutInterval = 2
        let (data, _) = try await URLSession.shared.data(for: req)
        self.info = try JSONDecoder().decode(Info.self, from: data)
        try await Self.post(baseURL.appending(path: "reset"), body: Data("{}".utf8))
    }

    public mutating func chooseAction(game: SnakeGame, snakeID: Int) async throws -> Direction {
        let features = FeatureExtractor.features(of: game, for: snakeID)
        let body = try JSONEncoder().encode(["features": features])
        let data = try await Self.post(baseURL.appending(path: "act"), body: body)
        let resp = try JSONDecoder().decode(ActResponse.self, from: data)
        lastInferenceMs = resp.ms
        return predictedDirection(fromLogits: resp.logits, game: game, snakeID: snakeID)
    }

    @discardableResult
    private static func post(_ url: URL, body: Data) async throws -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.httpBody = body
        req.timeoutInterval = 2
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw ModelError.server(String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)")
        }
        return data
    }
}
