import Foundation
import LLMCoreAI

// Milestone 1: report which generator this build can use. Prompting,
// streaming and `--json` bench output arrive with llm_ai.serve.

print("CoreAI.framework in SDK: \(hasCoreAIFramework ? "yes" : "no (needs Xcode 27 / macOS 27)")")

do {
    let remote = try await RemoteGenerator()
    print("remote server: \(remote.label), max_seq_len=\(remote.info.maxSeqLen)")
} catch {
    print("remote server: not reachable on :8770 — start `python -m llm_ai.serve` (\(error))")
}
