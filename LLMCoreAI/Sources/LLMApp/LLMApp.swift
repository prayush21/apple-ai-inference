import SwiftUI
import LLMCoreAI

// SmolLM2 chat box with a tokens/sec gauge — the LLM counterpart of
// SnakeApp's inference-ms HUD. Runs the asset in-process on CoreAI.framework
// (macOS 27). Prompts come pre-tokenized from data/llm/prompt_ids.json until
// a Swift BPE encoder exists, so the prompt is a picker, not a text field.
//
// Paths default to the repo layout relative to LLMCoreAI/ (the directory
// `swift run LLMApp` is started from); override with LLM_MODEL, LLM_TOKENIZER
// and LLM_PROMPTS. LLM_AUTOGENERATE=1 runs the first prompt after loading;
// LLM_SNAPSHOT=<path.png> then renders the window's content to that file.

@main
struct LLMApp: App {
    var body: some Scene {
        WindowGroup("SmolLM2 · Core AI") {
            if #available(macOS 27, *) {
                ChatView()
            } else {
                Text("CoreAI.framework needs macOS 27 — use llm_ai.serve").padding(32)
            }
        }
    }
}

enum Paths {
    static func url(_ env: String, _ fallback: String) -> URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment[env] ?? fallback)
    }
    static let model = url("LLM_MODEL", "../models/llm/SmolLM2Stateful.aimodel")
    static let tokenizer = url("LLM_TOKENIZER", "../models/llm/hf/SmolLM2-360M-Instruct/tokenizer.json")
    static let prompts = url("LLM_PROMPTS", "../data/llm/prompt_ids.json")
}

@available(macOS 27, *)
@MainActor
final class ChatModel: ObservableObject {
    @Published var status = "Loading model…"
    @Published var prompts: [PromptIDs.Chat] = []
    @Published var selected = 0
    @Published var reply = ""
    @Published var running = false
    @Published var loadMS: Double?
    @Published var firstTokenMS: Double?
    @Published var tokensPerSecond: Double = 0
    @Published var tokenCount = 0
    private var generator: ModelGenerator?

    func load() async {
        do {
            let ids = try PromptIDs.load(Paths.prompts)
            prompts = ids.chat
            let g = try await ModelGenerator(modelURL: Paths.model, tokenizerURL: Paths.tokenizer, prompts: ids)
            generator = g
            loadMS = g.loadMS
            status = "\(g.precision) · prefill t64 + decode · \(g.maxContext)-token cache"
            // For unattended screenshots: generate once as soon as the model is loaded.
            let env = ProcessInfo.processInfo.environment
            if env["LLM_AUTOGENERATE"] == "1" {
                await generate()
                if let path = env["LLM_SNAPSHOT"] { snapshot(to: URL(fileURLWithPath: path)) }
            }
        } catch {
            status = "Load failed: \(error)"
        }
    }

    func generate() async {
        guard let generator, prompts.indices.contains(selected) else { return }
        running = true
        reply = ""
        firstTokenMS = nil
        tokenCount = 0
        tokensPerSecond = 0
        defer { running = false }
        do {
            try await generator.reset()
            var decodeMS = 0.0
            for try await t in generator.generate(prompt: prompts[selected].prompt, maxTokens: 256) {
                reply += t.text
                if tokenCount == 0 {
                    firstTokenMS = t.ms
                } else {
                    decodeMS += t.ms
                    tokensPerSecond = Double(tokenCount) / decodeMS * 1e3
                }
                tokenCount += 1
            }
        } catch {
            reply += "\n[\(error)]"
        }
    }
}

@available(macOS 27, *)
extension ChatModel {
    /// Render the chat content (same view as the window) to a PNG.
    func snapshot(to url: URL) {
        let renderer = ImageRenderer(content: ChatContent(model: self, snapshot: true)
            .frame(width: 720, height: 560)
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
            .frame(minWidth: 640, minHeight: 480)
            .task { await model.load() }
    }
}

@available(macOS 27, *)
struct ChatContent: View {
    @ObservedObject var model: ChatModel
    /// ImageRenderer cannot draw AppKit-backed controls (Picker, Button,
    /// ScrollView), so the snapshot swaps them for plain text.
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
                Text("> " + (model.prompts.indices.contains(model.selected) ? model.prompts[model.selected].prompt : ""))
                    .font(.headline)
                Text(model.reply)
                    .font(.body.monospaced())
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(8)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            } else {
            HStack {
                Picker("Prompt", selection: $model.selected) {
                    ForEach(model.prompts.indices, id: \.self) { i in Text(model.prompts[i].prompt).tag(i) }
                }
                Button(model.running ? "Generating…" : "Generate") { Task { await model.generate() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.running || model.loadMS == nil)
            }

            ScrollView {
                Text(model.reply.isEmpty ? " " : model.reply)
                    .font(.body.monospaced())
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .defaultScrollAnchor(.top)
            .frame(minHeight: 220)
            .padding(8)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
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
                stat("load", model.loadMS.map { String(format: "%.1f s", $0 / 1e3) } ?? "—")
                Spacer()
            }
        }
        .padding(20)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading) {
            Text(value).font(.title3.monospacedDigit())
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
    }
}
