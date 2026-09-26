import Foundation

/// SmolLM2-Instruct's chat template (`tokenizer_config.json`), spelled out;
/// matches `llm_ai.tokenizer.chat_prompt`.
public enum ChatTemplate {
    public static let defaultSystem = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    public static let start = "<|im_start|>"
    public static let end = "<|im_end|>"

    /// System message + first user turn + the assistant header.
    public static func opening(system: String? = nil, user: String) -> String {
        "\(start)system\n\(system ?? defaultSystem)\(end)\n" + turn(user: user)
    }

    /// One more user turn + the assistant header.
    public static func turn(user: String) -> String {
        "\(start)user\n\(user)\(end)\n\(start)assistant\n"
    }
}
