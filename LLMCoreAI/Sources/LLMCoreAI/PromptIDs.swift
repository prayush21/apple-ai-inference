import Foundation

/// `data/llm/prompt_ids.json` from `python -m llm_ai.prompt_ids`: chat
/// prompts with the template applied and tokenized, the bench inputs, and a
/// greedy reference for the first chat prompt.
public struct PromptIDs: Decodable, Sendable {
    public struct Chat: Decodable, Sendable {
        public let prompt: String
        public let ids: [Int]
    }
    public struct Reference: Decodable, Sendable {
        public let chatIndex: Int
        public let greedyIds: [Int]
        public let text: String
    }

    public let tokenizer: String
    public let eosId: Int
    public let padId: Int
    public let chat: [Chat]
    public let bench: [String: [Int]]
    public let reference: Reference

    public static func load(_ url: URL) throws -> PromptIDs {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(PromptIDs.self, from: Data(contentsOf: url))
    }

    /// Token ids for a chat prompt, if it was pre-tokenized.
    public func ids(for prompt: String) -> [Int]? {
        chat.first { $0.prompt == prompt }?.ids
    }
}
