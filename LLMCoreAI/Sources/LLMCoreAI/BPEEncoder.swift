import Foundation

/// Text -> token ids for SmolLM2's byte-level BPE, read from the Hugging Face
/// `tokenizer.json`. Reproduces `tokenizers` for this checkpoint (checked
/// against it on a few hundred strings by `LLMCoreAITests`):
///
/// 1. **Special tokens** (`added_tokens`, e.g. `<|im_start|>`) are split out
///    first and map straight to their ids.
/// 2. **Digits** pre-tokenizer with `individual_digits: true`: every numeric
///    character becomes its own piece.
/// 3. **ByteLevel** pre-tokenizer (`add_prefix_space: false`) splits each
///    piece with GPT-2's pattern, matched on Unicode scalars:
///    `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
///    and maps every UTF-8 byte to GPT-2's printable character.
/// 4. **BPE**: merge the adjacent pair with the lowest rank until none
///    applies, then look each symbol up in the vocabulary.
public final class BPEEncoder: @unchecked Sendable {
    private let vocab: [String: Int]
    private let ranks: [Pair: Int]
    private let special: [(text: String, id: Int)]
    private let byteToUnicode: [Character]
    private var cache: [String: [Int]] = [:]
    private let lock = NSLock()

    private struct Pair: Hashable {
        let a: String, b: String
    }

    public init(tokenizerJSON url: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        guard let model = json?["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else {
            throw GeneratorError.server("\(url.lastPathComponent): no model.vocab / model.merges")
        }
        self.vocab = vocab
        var ranks: [Pair: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, m) in merges.enumerated() {
            // Merges are "a b" strings in older files and ["a", "b"] arrays in newer ones.
            let parts: [String]
            if let s = m as? String {
                parts = s.split(separator: " ", maxSplits: 1).map(String.init)
            } else {
                parts = m as? [String] ?? []
            }
            if parts.count == 2 { ranks[Pair(a: parts[0], b: parts[1])] = rank }
        }
        self.ranks = ranks
        let added = (json?["added_tokens"] as? [[String: Any]] ?? []).compactMap { t -> (String, Int)? in
            guard let c = t["content"] as? String, let id = t["id"] as? Int else { return nil }
            return (c, id)
        }
        // Longest first, so a special token that is a prefix of another never wins.
        self.special = added.sorted { $0.0.count > $1.0.count }
        let map = ByteLevelDecoder.unicodeToByte()
        var table = [Character](repeating: " ", count: 256)
        for (scalar, byte) in map { table[Int(byte)] = Character(scalar) }
        self.byteToUnicode = table
    }

    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        for segment in splitSpecial(text) {
            switch segment {
            case .special(let id):
                ids.append(id)
            case .text(let s):
                for piece in Self.splitDigits(s) {
                    for word in Self.splitGPT2(piece) {
                        ids += bpe(word)
                    }
                }
            }
        }
        return ids
    }

    // MARK: 1. special tokens

    private enum Segment {
        case special(Int)
        case text(String)
    }

    private func splitSpecial(_ text: String) -> [Segment] {
        var out: [Segment] = []
        var rest = Substring(text)
        var plain = ""
        while !rest.isEmpty {
            if let hit = special.first(where: { rest.hasPrefix($0.text) }) {
                if !plain.isEmpty { out.append(.text(plain)); plain = "" }
                out.append(.special(hit.id))
                rest = rest.dropFirst(hit.text.count)
            } else {
                plain.append(rest.removeFirst())
            }
        }
        if !plain.isEmpty { out.append(.text(plain)) }
        return out
    }

    // MARK: 2. digits

    static func isNumeric(_ s: Unicode.Scalar) -> Bool {
        switch s.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }

    /// Every numeric scalar isolated into its own piece.
    static func splitDigits(_ s: String) -> [String] {
        var out: [String] = [], current = String.UnicodeScalarView()
        for u in s.unicodeScalars {
            if isNumeric(u) {
                if !current.isEmpty { out.append(String(current)); current = .init() }
                out.append(String(u))
            } else {
                current.append(u)
            }
        }
        if !current.isEmpty { out.append(String(current)) }
        return out
    }

    // MARK: 3. GPT-2 pre-tokenizer, spelled out on Unicode scalars

    static func isLetter(_ u: Unicode.Scalar) -> Bool {
        switch u.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }

    static func isSpace(_ u: Unicode.Scalar) -> Bool { u.properties.isWhitespace }

    static func splitGPT2(_ s: String) -> [String] {
        let u = Array(s.unicodeScalars)
        var out: [String] = []
        var i = 0
        func emit(_ from: Int, _ to: Int) {
            var v = String.UnicodeScalarView()
            v.append(contentsOf: u[from..<to])
            out.append(String(v))
        }
        while i < u.count {
            // 's 't 're 've 'm 'll 'd
            if u[i] == "'" {
                let rest = String(String.UnicodeScalarView(u[(i + 1)...].prefix(2)))
                if let c = ["re", "ve", "ll"].first(where: { rest.hasPrefix($0) }) {
                    emit(i, i + 1 + c.unicodeScalars.count); i += 1 + c.unicodeScalars.count; continue
                }
                if let c = ["s", "t", "m", "d"].first(where: { rest.hasPrefix($0) }) {
                    emit(i, i + 1 + c.unicodeScalars.count); i += 1 + c.unicodeScalars.count; continue
                }
            }
            // optional single U+0020, then a run of one class
            let start = i
            var j = i
            if u[j] == " ", j + 1 < u.count, !isSpace(u[j + 1]) { j += 1 }
            let c = u[j]
            if isLetter(c) || isNumeric(c) || !isSpace(c) {
                let cls: (Unicode.Scalar) -> Bool
                if isLetter(c) {
                    cls = isLetter
                } else if isNumeric(c) {
                    cls = isNumeric
                } else {
                    cls = { !isSpace($0) && !isLetter($0) && !isNumeric($0) }
                }
                var k = j + 1
                while k < u.count, cls(u[k]) { k += 1 }
                emit(start, k); i = k; continue
            }
            // whitespace: \s+(?!\S) keeps the last space for the next word; else \s+
            var k = i
            while k < u.count, isSpace(u[k]) { k += 1 }
            if k < u.count, k - i > 1 { k -= 1 }
            emit(i, k); i = k
        }
        return out
    }

    // MARK: 4. BPE

    private func bpe(_ word: String) -> [Int] {
        lock.lock()
        if let hit = cache[word] { lock.unlock(); return hit }
        lock.unlock()
        var symbols = Array(word.utf8).map { String(byteToUnicode[Int($0)]) }
        while symbols.count > 1 {
            var best: (rank: Int, index: Int)?
            for i in 0..<(symbols.count - 1) {
                if let r = ranks[Pair(a: symbols[i], b: symbols[i + 1])], r < (best?.rank ?? .max) {
                    best = (r, i)
                }
            }
            guard let (_, i) = best else { break }
            let merged = symbols[i] + symbols[i + 1]
            // Merge every occurrence of this pair, left to right.
            let (a, b) = (symbols[i], symbols[i + 1])
            var next: [String] = [], k = 0
            while k < symbols.count {
                if k < symbols.count - 1, symbols[k] == a, symbols[k + 1] == b {
                    next.append(merged); k += 2
                } else {
                    next.append(symbols[k]); k += 1
                }
            }
            symbols = next
        }
        let ids = symbols.compactMap { vocab[$0] }
        lock.lock()
        cache[word] = ids
        lock.unlock()
        return ids
    }
}
