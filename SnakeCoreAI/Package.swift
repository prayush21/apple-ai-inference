// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SnakeCoreAI",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "SnakeEngine", targets: ["SnakeEngine"]),
        .library(name: "SnakeCoreAI", targets: ["SnakeCoreAI"]),
        .executable(name: "snake-cli", targets: ["snake-cli"]),
        .executable(name: "SnakeApp", targets: ["SnakeApp"]),
    ],
    targets: [
        // Pure-Swift game rules + feature extraction. Mirrors snake_ai/game.py
        // and snake_ai/features.py exactly.
        .target(name: "SnakeEngine"),

        // Players. `ModelPlayer` uses the CoreAI framework and is compiled only
        // where the SDK provides it (Xcode 27 / macOS 27+).
        .target(name: "SnakeCoreAI", dependencies: ["SnakeEngine"]),

        .executableTarget(name: "snake-cli", dependencies: ["SnakeEngine", "SnakeCoreAI"]),

        // SwiftUI game: arrow keys drive snake B, the model (or heuristic
        // fallback) drives snake A.
        .executableTarget(name: "SnakeApp", dependencies: ["SnakeEngine", "SnakeCoreAI"]),

        .testTarget(
            name: "SnakeEngineTests",
            dependencies: ["SnakeEngine"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
