// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DecideCoreAI",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "DecideCoreAI", targets: ["DecideCoreAI"]),
        .executable(name: "decide-cli", targets: ["decide-cli"]),
    ],
    targets: [
        // Jev-shaped request/response types, `RemoteDecider` (HTTP to
        // decide_ai.serve, works today) and `Decider` on CoreAI.framework
        // (compiled only under Xcode 27 / macOS 27).
        .target(name: "DecideCoreAI"),
        .executableTarget(name: "decide-cli", dependencies: ["DecideCoreAI"]),
    ]
)
