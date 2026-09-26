import Foundation
import Testing
@testable import LLMCoreAI

/// The Swift encoder must reproduce Hugging Face `tokenizers` exactly
/// (fixture: scripts/llm_tokenizer_fixture.py). Needs the downloaded
/// tokenizer.json; loads no model, so no specialization cache is written.
struct BPEEncoderTests {
    static let tokenizerURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appending(path: "../../../models/llm/hf/SmolLM2-360M-Instruct/tokenizer.json")
        .standardized

    struct Case: Decodable {
        let text: String
        let ids: [Int]
    }

    @Test(.enabled(if: FileManager.default.fileExists(atPath: tokenizerURL.path)))
    func matchesHuggingFaceTokenizers() throws {
        let url = try #require(Bundle.module.url(forResource: "encode", withExtension: "json", subdirectory: "Fixtures"))
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: url))
        let encoder = try BPEEncoder(tokenizerJSON: Self.tokenizerURL)
        let decoder = try ByteLevelDecoder(tokenizerJSON: Self.tokenizerURL)
        var failures = 0
        for c in cases {
            let got = encoder.encode(c.text)
            if got != c.ids {
                failures += 1
                if failures <= 10 { Issue.record("\(c.text.debugDescription)\n  got      \(got)\n  expected \(c.ids)") }
            }
            #expect(decoder.decode(c.ids) == c.text, "decode round trip: \(c.text.debugDescription)")
        }
        #expect(failures == 0, "\(failures)/\(cases.count) strings encode differently")
    }

    /// Chat template + encoder in Swift == `llm_ai.tokenizer.encode_chat` in Python.
    @Test(.enabled(if: FileManager.default.fileExists(atPath: tokenizerURL.path)))
    func chatPromptsMatchPython() throws {
        let promptsURL = Self.tokenizerURL.appending(path: "../../../../../data/llm/prompt_ids.json").standardized
        let prompts = try PromptIDs.load(promptsURL)
        let encoder = try BPEEncoder(tokenizerJSON: Self.tokenizerURL)
        for chat in prompts.chat {
            #expect(encoder.encode(ChatTemplate.opening(user: chat.prompt)) == chat.ids, "\(chat.prompt)")
        }
    }
}
