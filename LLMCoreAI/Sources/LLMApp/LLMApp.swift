import SwiftUI
import LLMCoreAI

// Placeholder window. The real view — prompt field, streamed transcript,
// tokens/sec + load-ms gauges, generator picker — lands once RemoteGenerator
// streams.

@main
struct LLMApp: App {
    var body: some Scene {
        WindowGroup("SmolLM2 · Core AI") {
            VStack(spacing: 12) {
                Text("SmolLM2-360M on Core AI").font(.title2)
                Text(hasCoreAIFramework
                     ? "CoreAI.framework available — in-process generator"
                     : "CoreAI.framework needs macOS 27 — will use llm_ai.serve")
                    .foregroundStyle(.secondary)
            }
            .padding(32)
            .frame(minWidth: 480, minHeight: 240)
        }
    }
}
