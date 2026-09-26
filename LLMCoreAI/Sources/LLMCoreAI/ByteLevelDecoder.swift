import Foundation

/// Token ids -> text for SmolLM2's byte-level BPE, read straight from the
/// Hugging Face `tokenizer.json`. Decoding needs no merges: every vocabulary
/// entry is a string of GPT-2 "printable" characters, one per byte, so the
/// bytes come back through the inverse of `bytes_to_unicode` and the
/// concatenation is UTF-8. Special tokens (`<|im_end|>`, …) decode to their
/// literal text, like `tokenizers.decode(skip_special_tokens=False)`.
///
/// Encoding (the BPE merges and the pre-tokenizer regex) is not ported yet;
/// prompts arrive pre-tokenized from `python -m llm_ai.prompt_ids`.
public struct ByteLevelDecoder: Sendable {
    private let tokenBytes: [[UInt8]]

    public init(tokenizerJSON url: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        guard let model = json?["model"] as? [String: Any], let vocab = model["vocab"] as? [String: Int] else {
            throw GeneratorError.server("\(url.lastPathComponent): no model.vocab")
        }
        let unicodeToByte = Self.unicodeToByte()
        var table = [[UInt8]](repeating: [], count: (vocab.values.max() ?? -1) + 1)
        for (token, id) in vocab {
            table[id] = token.unicodeScalars.compactMap { unicodeToByte[$0] }
        }
        // Special tokens are stored verbatim, not through the byte map.
        for added in json?["added_tokens"] as? [[String: Any]] ?? [] {
            if let id = added["id"] as? Int, let content = added["content"] as? String, id < table.count {
                table[id] = Array(content.utf8)
            }
        }
        self.tokenBytes = table
    }

    public var vocabularySize: Int { tokenBytes.count }

    /// UTF-8 bytes of `ids`, concatenated.
    public func bytes(_ ids: some Sequence<Int>) -> [UInt8] {
        ids.flatMap { $0 >= 0 && $0 < tokenBytes.count ? tokenBytes[$0] : [] }
    }

    /// Text of `ids`. A multi-byte character split across the last token
    /// comes out as U+FFFD; decode the whole reply so far when streaming.
    public func decode(_ ids: some Sequence<Int>) -> String {
        String(decoding: bytes(ids), as: UTF8.self)
    }

    /// GPT-2's `bytes_to_unicode`, inverted: printable Latin-1 bytes map to
    /// themselves, the other 68 bytes to U+0100 onwards in byte order.
    static func unicodeToByte() -> [Unicode.Scalar: UInt8] {
        let printable = Array(33...126) + Array(161...172) + Array(174...255)
        var map: [Unicode.Scalar: UInt8] = [:]
        var extra: UInt32 = 0
        for b in 0...255 {
            let scalar: Unicode.Scalar
            if printable.contains(b) {
                scalar = Unicode.Scalar(UInt32(b))!
            } else {
                scalar = Unicode.Scalar(256 + extra)!
                extra += 1
            }
            map[scalar] = UInt8(b)
        }
        return map
    }
}
