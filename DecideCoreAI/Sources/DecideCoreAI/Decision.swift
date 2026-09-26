import Foundation

// The wire format is Jev's `/v1/evaluate` (Vercel AI Gateway), which
// `decide_ai.serve` mirrors on `/decide`, so one client speaks to both.

public struct Question: Codable, Sendable, Equatable {
    public var type: String
    /// Question form, e.g. "Is the customer asking for a refund?" (what Jev is sent).
    public var instructions: String
    /// Declarative NLI form, e.g. "The customer is asking for a refund." (what the local model scores).
    public var hypothesis: String?

    public init(_ instructions: String, hypothesis: String? = nil) {
        self.type = "boolean"
        self.instructions = instructions
        self.hypothesis = hypothesis
    }

    /// `type` defaults to boolean so the question files (which omit it) decode.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decodeIfPresent(String.self, forKey: .type) ?? "boolean"
        instructions = try c.decode(String.self, forKey: .instructions)
        hypothesis = try c.decodeIfPresent(String.self, forKey: .hypothesis)
    }

    enum CodingKeys: String, CodingKey { case type, instructions, hypothesis }
}

public struct DecideRequest: Codable, Sendable {
    public var model: String
    public var state: String
    public var questions: [String: Question]
    /// `decide_ai.serve` only: pin the padded sequence length (bench).
    public var paddedLen: Int?

    enum CodingKeys: String, CodingKey { case model, state, questions, paddedLen = "padded_len" }

    public init(model: String = "local/nli-minilm2", state: String, questions: [String: Question], paddedLen: Int? = nil) {
        self.model = model
        self.state = state
        self.questions = questions
        self.paddedLen = paddedLen
    }
}

public struct Answer: Codable, Sendable, Equatable {
    public var type: String
    public var probability: Double
}

public struct Usage: Codable, Sendable, Equatable {
    public var inputTokens: Int
    public var outputTokens: Int
}

/// Server-side timing; only `decide_ai.serve` sends it. The Laya backend
/// (`--backend laya`) reports `ms_total` and `batch` only.
public struct Timing: Codable, Sendable, Equatable {
    public var msTokenize: Double?
    public var msInfer: Double?
    public var msTotal: Double
    public var batch: Int
    public var paddedLen: Int?

    enum CodingKeys: String, CodingKey {
        case msTokenize = "ms_tokenize", msInfer = "ms_infer", msTotal = "ms_total", batch, paddedLen = "padded_len"
    }
}

public struct DecideResponse: Codable, Sendable {
    public var model: String
    public var answers: [String: Answer]
    public var usage: Usage?
    public var timing: Timing?
    /// `[P(contradiction), P(entailment), P(neutral)]` per question, local server only.
    /// The Laya server sends `{noul, act_probability, temperature}` objects instead; those decode as nil.
    public var raw: [String: [Double]]?

    public init(model: String, answers: [String: Answer], usage: Usage? = nil, timing: Timing? = nil,
                raw: [String: [Double]]? = nil) {
        self.model = model
        self.answers = answers
        self.usage = usage
        self.timing = timing
        self.raw = raw
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decode(String.self, forKey: .model)
        answers = try c.decode([String: Answer].self, forKey: .answers)
        usage = try c.decodeIfPresent(Usage.self, forKey: .usage)
        timing = try c.decodeIfPresent(Timing.self, forKey: .timing)
        raw = try? c.decodeIfPresent([String: [Double]].self, forKey: .raw)
    }

    enum CodingKeys: String, CodingKey { case model, answers, usage, timing, raw }
}

public enum DeciderError: Error, CustomStringConvertible {
    case coreAIUnavailable
    case missingFunction(String)
    case missingOutput(String)
    case server(String)
    case referenceMismatch(Double)

    public var description: String {
        switch self {
        case .coreAIUnavailable: "CoreAI.framework is not available in this SDK (needs macOS 27 / Xcode 27)"
        case .missingFunction(let n): "model has no function named \(n)"
        case .missingOutput(let n): "model returned no output named \(n)"
        case .server(let m): "server error: \(m)"
        case .referenceMismatch(let d): "logits differ from the Python Core AI reference by \(d) (regenerate with python -m decide_ai.bench_ids)"
        }
    }
}

/// The five triage questions used by the holdout, the bench and the README.
public enum TriageQuestions {
    public static let all: [String: Question] = [
        "is_complaint": Question("Is the customer complaining?", hypothesis: "The customer is complaining."),
        "wants_refund": Question("Is the customer asking for a refund?", hypothesis: "The customer is asking for a refund."),
        "about_shipping": Question("Is this message about shipping or delivery?", hypothesis: "This message is about shipping or delivery."),
        "about_product_quality": Question("Is this message about product quality?", hypothesis: "This message is about product quality."),
        "urgent": Question("Is this urgent?", hypothesis: "This is urgent."),
    ]
}
