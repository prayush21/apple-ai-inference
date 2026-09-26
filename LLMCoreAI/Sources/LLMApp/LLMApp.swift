import AppKit
import SwiftUI
import LLMCoreAI

// SmolLM2 chat with a tokens/sec gauge — the LLM counterpart of SnakeApp's
// inference-ms HUD. The model runs in-process on CoreAI.framework (macOS 27);
// text is tokenized in Swift and the conversation lives in the KV cache, so
// each message only prefills its own tokens. The cache holds 1024 tokens;
// "New chat" clears it.
//
// Paths default to the repo layout relative to LLMCoreAI/ (the directory
// `swift run LLMApp` is started from); override with LLM_MODEL and
// LLM_TOKENIZER. For unattended screenshots: LLM_AUTOGENERATE="<message>"
// sends that message after loading, and LLM_SNAPSHOT=<path.png> then renders
// the window's content to that file.

/// A SwiftPM executable is not an .app bundle, so macOS launches it as a
/// background process: the window shows, but keystrokes stay with the
/// terminal. Registering as a regular app and activating fixes typing.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate()
        NSApplication.shared.windows.first?.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct LLMApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("SmolLM2 · Core AI") {
            #if canImport(CoreAI)
            if #available(macOS 27, *) {
                ChatView()
            } else {
                Text("CoreAI.framework needs macOS 27 — use llm_ai.serve").padding(32)
            }
            #else
            // ModelGenerator is compiled out of LLMCoreAI without the macOS 27 SDK.
            Text("This build has no CoreAI.framework (needs Xcode 27) — use llm_ai.serve").padding(32)
            #endif
        }
    }
}

enum Paths {
    static func url(_ env: String, _ fallback: String) -> URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment[env] ?? fallback)
    }
    static let model = url("LLM_MODEL", "../models/llm/SmolLM2Stateful.aimodel")
    static let tokenizer = url("LLM_TOKENIZER", "../models/llm/hf/SmolLM2-360M-Instruct/tokenizer.json")
}

struct Turn: Identifiable {
    let id = UUID()
    let isUser: Bool
    var text: String
}

#if canImport(CoreAI)
@available(macOS 27, *)
@MainActor
final class ChatModel: ObservableObject {
    @Published var status = "Loading model… (the first launch specializes it: ~20–30 s)"
    @Published var turns: [Turn] = []
    @Published var draft = ""
    @Published var running = false
    @Published var loadMS: Double?
    @Published var firstTokenMS: Double?
    @Published var tokensPerSecond: Double = 0
    @Published var tokenCount = 0
    @Published var contextUsed = 0
    @Published var maxContext = 1024
    private var generator: ModelGenerator?
    private var task: Task<Void, Never>?

    var ready: Bool { generator != nil }

    func load() async {
        do {
            let g = try await ModelGenerator(modelURL: Paths.model, tokenizerURL: Paths.tokenizer)
            generator = g
            loadMS = g.loadMS
            maxContext = g.maxContext
            status = "\(g.precision) · prefill t64 + decode · GPU"
            let env = ProcessInfo.processInfo.environment
            if let message = env["LLM_AUTOGENERATE"] {
                for m in message.components(separatedBy: " || ") {
                    draft = m == "1" ? "Give me three tips for writing clear commit messages." : m
                    send()
                    await task?.value
                }
                if let path = env["LLM_SNAPSHOT"] { snapshot(to: URL(fileURLWithPath: path)) }
            }
        } catch {
            status = "Load failed: \(error)"
        }
    }

    func send() {
        let message = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let generator, !message.isEmpty, !running else { return }
        draft = ""
        turns.append(Turn(isUser: true, text: message))
        turns.append(Turn(isUser: false, text: ""))
        let index = turns.count - 1
        running = true
        firstTokenMS = nil
        tokenCount = 0
        tokensPerSecond = 0
        task = Task {
            var decodeMS = 0.0
            do {
                for try await t in generator.chat(message, maxTokens: 512) {
                    turns[index].text += t.text
                    if tokenCount == 0 {
                        firstTokenMS = t.ms
                    } else {
                        decodeMS += t.ms
                        tokensPerSecond = Double(tokenCount) / decodeMS * 1e3
                    }
                    tokenCount += 1
                }
            } catch GeneratorError.contextExhausted {
                turns[index].text += "\n[context full (\(maxContext) tokens) — start a new chat]"
            } catch {
                turns[index].text += "\n[\(error)]"
            }
            contextUsed = await generator.position
            running = false
        }
    }

    func stop() { task?.cancel() }

    func newChat() {
        guard let generator, !running else { return }
        Task {
            try? await generator.reset()
            turns = []
            contextUsed = 0
            firstTokenMS = nil
            tokenCount = 0
            tokensPerSecond = 0
        }
    }

    /// Render the chat content (same view as the window) to a PNG.
    func snapshot(to url: URL) {
        let renderer = ImageRenderer(content: ChatContent(model: self, snapshot: true)
            .frame(width: 720, height: 620)
            .background(Color(nsColor: .windowBackgroundColor)))
        renderer.scale = 2
        guard let image = renderer.nsImage, let tiff = image.tiffRepresentation,
              let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else { return }
        try? png.write(to: url)
    }
}

@available(macOS 27, *)
struct ChatView: View {
    @StateObject private var model = ChatModel()

    var body: some View {
        ChatContent(model: model)
            .frame(minWidth: 640, minHeight: 520)
            .task { await model.load() }
    }
}

@available(macOS 27, *)
struct ChatContent: View {
    @ObservedObject var model: ChatModel
    /// ImageRenderer cannot draw AppKit-backed controls (TextField, Button,
    /// ScrollView), so the snapshot swaps them for plain views.
    var snapshot = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("SmolLM2-360M-Instruct").font(.title2.bold())
                Spacer()
                Text("CoreAI.framework, in-process").foregroundStyle(.secondary)
            }
            Text(model.status).font(.caption).foregroundStyle(.secondary)

            if snapshot {
                transcript.frame(maxHeight: .infinity, alignment: .top)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        transcript
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .onChange(of: model.turns.last?.text) { proxy.scrollTo("bottom", anchor: .bottom) }
                }
                .frame(maxHeight: .infinity)

                HStack {
                    TextField(model.ready ? "Message SmolLM2" : "Loading…", text: $model.draft, axis: .vertical)
                        .lineLimit(1...5)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { model.send() }
                        .disabled(!model.ready)
                    if model.running {
                        Button("Stop") { model.stop() }
                    } else {
                        Button("Send") { model.send() }
                            .keyboardShortcut(.defaultAction)
                            .disabled(!model.ready || model.draft.isEmpty)
                    }
                    Button("New chat") { model.newChat() }
                        .disabled(model.running || model.turns.isEmpty)
                }
            }

            HStack(spacing: 24) {
                Gauge(value: min(model.tokensPerSecond, 100), in: 0...100) {
                    Text("tok/s")
                } currentValueLabel: {
                    Text(String(format: "%.1f", model.tokensPerSecond))
                }
                .gaugeStyle(.accessoryCircular)
                .tint(.green)
                stat("tokens", "\(model.tokenCount)")
                stat("first token", model.firstTokenMS.map { String(format: "%.0f ms", $0) } ?? "—")
                stat("context", "\(model.contextUsed) / \(model.maxContext)")
                stat("load", model.loadMS.map { String(format: "%.1f s", $0 / 1e3) } ?? "—")
                Spacer()
            }
        }
        .padding(20)
    }

    private var transcript: some View {
        VStack(alignment: .leading, spacing: 10) {
            if model.turns.isEmpty {
                Text(model.ready ? "Ask anything. Replies are greedy (deterministic)." : " ")
                    .foregroundStyle(.secondary)
            }
            ForEach(model.turns) { turn in
                HStack {
                    if turn.isUser { Spacer(minLength: 60) }
                    Text(turn.text.isEmpty ? "…" : turn.text)
                        .textSelection(.enabled)
                        .padding(10)
                        .background(turn.isUser ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12),
                                    in: RoundedRectangle(cornerRadius: 10))
                    if !turn.isUser { Spacer(minLength: 60) }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading) {
            Text(value).font(.title3.monospacedDigit())
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
    }
}
#endif
