import AppKit
import SnakeEngine
import SwiftUI

@main
struct SnakeApp: App {
    init() {
        // Running from `swift run` has no app bundle; make the process a
        // regular foreground app so the window gets focus and key events.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Core AI Snake") {
            ContentView()
        }
        .windowResizability(.contentSize)
    }
}

struct ContentView: View {
    @State private var vm = GameViewModel()
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 12) {
            picker
            hud
            BoardView(game: vm.game)
                .frame(width: 420, height: 420)
                .focusable()
                .focused($focused)
                .focusEffectDisabled()
                .onKeyPress(.upArrow) { vm.steer(.up); return .handled }
                .onKeyPress(.downArrow) { vm.steer(.down); return .handled }
                .onKeyPress(.leftArrow) { vm.steer(.left); return .handled }
                .onKeyPress(.rightArrow) { vm.steer(.right); return .handled }
                .onKeyPress(.space) { vm.start(); return .handled }
            footer
            scoreboard
        }
        .padding(16)
        .onAppear {
            focused = true
            // SNAKE_AUTOSTART=1 starts a game immediately (used for smoke tests).
            if ProcessInfo.processInfo.environment["SNAKE_AUTOSTART"] != nil { vm.start() }
        }
    }

    private var picker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Snake A", selection: $vm.aiKind) {
                ForEach(AIKind.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Text(vm.aiKind.blurb)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var scoreboard: some View {
        HStack(spacing: 14) {
            ForEach(AIKind.allCases) { kind in
                let s = vm.scores[kind] ?? Scoreboard()
                HStack(spacing: 4) {
                    Text(kind.rawValue)
                        .fontWeight(kind == vm.aiKind ? .semibold : .regular)
                    Text("AI \(s.aiWins) · you \(s.humanWins)\(s.draws > 0 ? " · draw \(s.draws)" : "")")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
            }
            Spacer()
            if let avg = vm.averageInferenceMs {
                Text(String(format: "avg %.2f ms/move", avg))
                    .font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
        }
    }

    private var hud: some View {
        HStack {
            Label("A · \(vm.aiLabel)", systemImage: "cpu").foregroundStyle(.green)
            Spacer()
            Label("B · you (arrow keys)", systemImage: "person").foregroundStyle(.blue)
        }
        .font(.callout)
    }

    private var footer: some View {
        HStack(spacing: 16) {
            Button(vm.phase == .running ? "Restart" : "Start") { vm.start(); focused = true }
                .keyboardShortcut(.defaultAction)
            Text("step \(vm.game.stepCount)")
            Text("A \(vm.game.snakes[0].length) · B \(vm.game.snakes[1].length)")
            if let ms = vm.lastInferenceMs {
                Text(String(format: "AI %.2f ms", ms)).monospacedDigit()
            }
            Spacer()
            switch vm.phase {
            case .idle: Text("press Start or space").foregroundStyle(.secondary)
            case .loading: ProgressView().controlSize(.small)
            case .running: EmptyView()
            case .over(let s): Text(s).bold()
            }
        }
        .font(.callout)
    }
}

/// Draws the grid, food and both snakes with a Canvas.
struct BoardView: View {
    let game: SnakeGame

    var body: some View {
        Canvas { ctx, size in
            let cell = min(size.width / CGFloat(game.width), size.height / CGFloat(game.height))
            func rect(_ p: Point, inset: CGFloat = 1) -> CGRect {
                CGRect(x: CGFloat(p.x) * cell, y: CGFloat(p.y) * cell, width: cell, height: cell)
                    .insetBy(dx: inset, dy: inset)
            }
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black.opacity(0.85)))
            for y in 0..<game.height {
                for x in 0..<game.width {
                    ctx.fill(Path(rect(Point(x, y))), with: .color(.white.opacity(0.04)))
                }
            }
            if game.inBounds(game.food) {
                ctx.fill(Path(ellipseIn: rect(game.food, inset: cell * 0.2)), with: .color(.red))
            }
            for (i, s) in game.snakes.enumerated() {
                let base: Color = i == 0 ? .green : .blue
                for (k, p) in s.body.enumerated() where game.inBounds(p) {
                    let c = s.alive ? base.opacity(k == 0 ? 1 : 0.7) : .gray
                    ctx.fill(Path(roundedRect: rect(p), cornerRadius: k == 0 ? cell * 0.3 : cell * 0.15), with: .color(c))
                }
            }
        }
        .background(.black)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
