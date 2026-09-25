import Foundation

/// Runs the NLI decider through `python -m decide_ai.serve` (the Core AI
/// Python runtime) over local HTTP, until `CoreAI.framework` is available
/// in-process. Measures the round trip and exposes the server's own timing.
public struct RemoteDecider: Sendable {
    public struct Info: Decodable, Sendable {
        /// `local-static` / `local-dynamic` / `laya`.
        public let backend: String
        /// The `.aimodel` file for the local backends, `"laya"` for Laya (no asset: PyTorch on the CPU).
        public let asset: String
        public let functions: [String]
        public let maxLen: Int
        public let temperature: Double
        public let score: String
        public let labels: [String]

        enum CodingKeys: String, CodingKey { case backend, asset, functions, maxLen = "max_len", temperature, score, labels }

        /// Laya pads to its own sequence, so the bench has no `L` to pin.
        public var pinsPaddedLen: Bool { backend != "laya" }
    }

    public struct Result: Sendable {
        public let response: DecideResponse
        /// Wall-clock `decide` as the caller experiences it (JSON + HTTP + inference).
        public let roundtripMs: Double
        /// The server's inference-only time, for attributing the difference to the HTTP hop (nil for Laya).
        public var serverInferMs: Double? { response.timing?.msInfer }
        public var serverTotalMs: Double? { response.timing?.msTotal }
    }

    public let baseURL: URL
    public let info: Info

    public init(baseURL: URL = URL(string: "http://127.0.0.1:8770")!) async throws {
        self.baseURL = baseURL
        var req = URLRequest(url: baseURL.appending(path: "info"))
        req.timeoutInterval = 5
        let (data, _) = try await URLSession.shared.data(for: req)
        self.info = try JSONDecoder().decode(Info.self, from: data)
    }

    public func decide(state: String, questions: [String: Question], paddedLen: Int? = nil) async throws -> Result {
        let body = try JSONEncoder().encode(DecideRequest(state: state, questions: questions, paddedLen: paddedLen))
        var req = URLRequest(url: baseURL.appending(path: "decide"))
        req.httpMethod = "POST"
        req.httpBody = body
        req.timeoutInterval = 120
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let t0 = DispatchTime.now().uptimeNanoseconds
        let (data, response) = try await URLSession.shared.data(for: req)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw DeciderError.server(String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)")
        }
        return Result(response: try JSONDecoder().decode(DecideResponse.self, from: data), roundtripMs: ms)
    }
}
