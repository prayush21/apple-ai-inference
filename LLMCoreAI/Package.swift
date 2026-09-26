// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LLMCoreAI",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "LLMCoreAI", targets: ["LLMCoreAI"]),
        .executable(name: "llm-cli", targets: ["llm-cli"]),
        .executable(name: "LLMApp", targets: ["LLMApp"]),
    ],
    targets: [
        // Token generators. `ModelGenerator` runs the .aimodel in-process on
        // CoreAI.framework and is compiled only where the SDK provides it
        // (Xcode 27 / macOS 27+); `RemoteGenerator` streams tokens from
        // llm_ai.serve over HTTP until then.
        .target(name: "LLMCoreAI"),

        .executableTarget(name: "llm-cli", dependencies: ["LLMCoreAI"]),

        // SwiftUI chat box with a tokens/sec gauge — the LLM counterpart of
        // SnakeApp's inference-ms HUD.
        .executableTarget(name: "LLMApp", dependencies: ["LLMCoreAI"]),

        // Tokenizer tests only (no model load, so no specialization cache).
        .testTarget(name: "LLMCoreAITests", dependencies: ["LLMCoreAI"], resources: [.copy("Fixtures")]),
    ]
)
