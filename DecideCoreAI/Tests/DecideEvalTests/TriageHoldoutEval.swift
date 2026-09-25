#if canImport(Evaluations) && canImport(CoreAI)
import CoreAI
import DecideCoreAI
import Evaluations
import Foundation
import Testing

// The 150-state human-reviewed triage holdout on Xcode 27's Evaluations
// framework: the same numbers `python -m decide_ai.calibrate` writes to
// docs/bench/decide-quality.md, now as an Xcode evaluation report.
//
//   .venv/bin/python -m decide_ai.eval_export     # data/decide/eval/triage_holdout.json
//   cd DecideCoreAI && rm -f ../docs/bench/eval/*.xcevalresult && swift test --filter TriageHoldout --attachments-path ../docs/bench/eval
//
// One sample per (state, question) cell, 750 in all: Evaluations requires the
// expected value and the subject's output to be the same type, and a cell's
// target probability (true 1, false 0, "unsure" 0.5) vs the model's P(yes)
// keeps every metric per sample. Subjects: MiniLM in-process on
// CoreAI.framework (three scorings of the same logits) and Jev replayed from
// data/decide/jev_cache.jsonl (the answers `calibrate.py` recorded).

let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "../../..").standardizedFileURL

@available(macOS 27, *)
struct TriageCell: SampleProtocol {
    let id: Int
    let category: String
    let question: String
    let state: String
    let expected: Double?
    let paddedLen: Int
    let ids: [Int32]
    let jevKey: String
    var input: String { "\(question): \(state)" }
    /// "unsure" cells score only `hedging`; everything else ignores them, as in calibrate.py.
    var unsure: Bool { expected == 0.5 }
    var yes: Bool { expected == 1 }
}

@available(macOS 27, *)
enum Holdout {
    static let questions = ["is_complaint", "wants_refund", "about_shipping", "about_product_quality", "urgent"]
    static let padID: Int32 = 1

    static let cells: [TriageCell] = {
        struct File: Decodable {
            struct Sample: Decodable {
                let id: Int, category: String, input: String, expected: [String: Double]
                let padded_len: Int, ids: [[Int32]], jev_key: String
            }
            let questions: [String]
            let samples: [Sample]
        }
        let url = repo.appending(path: "data/decide/eval/triage_holdout.json")
        guard let data = try? Data(contentsOf: url), let f = try? JSONDecoder().decode(File.self, from: data) else {
            fatalError("missing \(url.path); run .venv/bin/python -m decide_ai.eval_export")
        }
        precondition(f.questions == questions)
        return f.samples.flatMap { s in
            questions.enumerated().map { qi, q in
                TriageCell(id: s.id, category: s.category, question: q, state: s.input, expected: s.expected[q],
                           paddedLen: s.padded_len, ids: s.ids[qi], jevKey: s.jev_key)
            }
        }
    }()

    /// sha256 of the canonical request -> {question: probability}
    static let jev: [String: [String: Double]] = {
        struct Rec: Decodable {
            struct Resp: Decodable { struct A: Decodable { let probability: Double }; let answers: [String: A] }
            let key: String, response: Resp
        }
        let text = (try? String(contentsOf: repo.appending(path: "data/decide/jev_cache.jsonl"), encoding: .utf8)) ?? ""
        var out: [String: [String: Double]] = [:]
        for line in text.split(separator: "\n") {
            if let r = try? JSONDecoder().decode(Rec.self, from: Data(line.utf8)) {
                out[r.key] = r.response.answers.mapValues(\.probability)  // last write wins, like jev.py
            }
        }
        return out
    }()
}

@available(macOS 27, *)
actor MiniLM {
    static let shared = MiniLM()
    private var decider: Decider?

    /// [contradiction, entailment, neutral] logits for one cell, dynamic asset, padded like calibrate.py.
    func logits(_ c: TriageCell) async throws -> [Float] {
        if decider == nil {
            decider = try await Decider(modelURL: repo.appending(path: "models/decide/NLICrossEncoder.aimodel"))
        }
        let pad = c.paddedLen - c.ids.count
        let ids = c.ids + Array(repeating: Holdout.padID, count: pad)
        let mask = Array(repeating: Int32(1), count: c.ids.count) + Array(repeating: 0, count: pad)
        return try await decider!.logits(inputIDs: [ids], attentionMask: [mask])[0]
    }
}

func softmax(_ x: [Float], temperature t: Double) -> [Double] {
    let s = x.map { Double($0) / t }, m = s.max()!
    let e = s.map { exp($0 - m) }, z = e.reduce(0, +)
    return e.map { $0 / z }
}

// MARK: - Metrics

/// AUC and ECE need every (p, y) pair at once, but `MetricsAggregator.custom`
/// receives one metric's values; so the pair is packed as 2y + p (p in [0, 1]).
func unpack(_ v: Double) -> (p: Double, y: Double) { v >= 2 ? (v - 2, 1) : (v, 0) }

func accuracy(_ xs: [Double]) -> Double {
    let pairs = xs.map(unpack)
    return Double(pairs.filter { ($0.p >= 0.5) == ($0.y == 1) }.count) / Double(pairs.count)
}

/// calibrate.py's ECE: confidence of the predicted class, 10 equal-width bins.
func ece(_ xs: [Double]) -> Double {
    var bins = Array(repeating: (n: 0.0, conf: 0.0, correct: 0.0), count: 10)
    for (p, y) in xs.map(unpack) {
        let pred = p >= 0.5, conf = pred ? p : 1 - p
        let b = min(Int(conf * 10), 9)
        bins[b].n += 1; bins[b].conf += conf; bins[b].correct += (pred == (y == 1)) ? 1 : 0
    }
    return bins.filter { $0.n > 0 }.reduce(0) { $0 + $1.n / Double(xs.count) * abs($1.correct / $1.n - $1.conf / $1.n) }
}

/// Mann-Whitney AUC with ties counted half, as calibrate.py.
func auc(_ xs: [Double]) -> Double {
    let pairs = xs.map(unpack)
    let pos = pairs.filter { $0.y == 1 }.map(\.p), neg = pairs.filter { $0.y == 0 }.map(\.p)
    guard !pos.isEmpty, !neg.isEmpty else { return .nan }
    var s = 0.0
    for a in pos { for b in neg { s += a > b ? 1 : (a == b ? 0.5 : 0) } }
    return s / Double(pos.count * neg.count)
}

func recall(_ xs: [Double]) -> Double {
    let pos = xs.map(unpack).filter { $0.y == 1 }
    return pos.isEmpty ? .nan : Double(pos.filter { $0.p >= 0.5 }.count) / Double(pos.count)
}

func yesRate(_ xs: [Double]) -> Double { Double(xs.map(unpack).filter { $0.p >= 0.5 }.count) / Double(xs.count) }

@available(macOS 27, *)
enum M {
    static let correct = Metric("correct")
    static let brier = Metric("brier")
    static let hedging = Metric("hedging")
    static let pair = Metric("pair (2y+p)")
    static let perQuestion = Dictionary(uniqueKeysWithValues: Holdout.questions.map { ($0, Metric("\($0) pair (2y+p)")) })
    static let perQuestionHedging = Dictionary(uniqueKeysWithValues: Holdout.questions.map { ($0, Metric("\($0) hedging")) })
}

// MARK: - Evaluations

@available(macOS 27, *)
protocol TriageEvaluation: Evaluation
where Sample == TriageCell, Subject == ModelSubject<Double>, SampleLoader == ArrayLoader<TriageCell> {
    func probability(_ cell: TriageCell) async throws -> Double
}

@available(macOS 27, *)
extension TriageEvaluation {
    var dataset: ArrayLoader<TriageCell> { ArrayLoader(samples: Holdout.cells) }

    func subject(from sample: TriageCell) async throws -> ModelSubject<Double> {
        ModelSubject(value: try await probability(sample))
    }

    var evaluators: [any EvaluatorProtocol<TriageCell, ModelSubject<Double>>] {
        var e: [any EvaluatorProtocol<TriageCell, ModelSubject<Double>>] = [
            Evaluator<TriageCell> { c, s in
                c.unsure ? M.correct.ignore() : M.correct.scoring((s.value >= 0.5) == c.yes ? 1 : 0)
            },
            Evaluator<TriageCell> { c, s in
                c.unsure ? M.brier.ignore() : M.brier.scoring((s.value - c.expected!) * (s.value - c.expected!))
            },
            Evaluator<TriageCell> { c, s in c.unsure ? M.hedging.scoring(abs(s.value - 0.5)) : M.hedging.ignore() },
            Evaluator<TriageCell> { c, s in c.unsure ? M.pair.ignore() : M.pair.scoring(2 * c.expected! + s.value) },
        ]
        for q in Holdout.questions {
            e.append(Evaluator<TriageCell> { c, s in
                c.question != q || c.unsure ? M.perQuestion[q]!.ignore() : M.perQuestion[q]!.scoring(2 * c.expected! + s.value)
            })
            e.append(Evaluator<TriageCell> { c, s in
                c.question == q && c.unsure ? M.perQuestionHedging[q]!.scoring(abs(s.value - 0.5)) : M.perQuestionHedging[q]!.ignore()
            })
        }
        return e
    }

    func aggregateMetrics(using a: inout MetricsAggregator) {
        a.computeMean(of: M.correct)
        a.computeMean(of: M.brier)
        a.computeMean(of: M.hedging)
        a.custom(of: M.pair, label: "ECE", ece)
        a.custom(of: M.pair, label: "AUC", auc)
        a.custom(of: M.pair, label: "yes-rate", yesRate)
        a.custom(of: M.pair, label: "recall", recall)
        for q in Holdout.questions {
            a.group(q) { g in
                g.custom(of: M.perQuestion[q]!, label: "\(q) accuracy", accuracy)
                g.custom(of: M.perQuestion[q]!, label: "\(q) ECE", ece)
                g.custom(of: M.perQuestion[q]!, label: "\(q) AUC", auc)
                g.custom(of: M.perQuestion[q]!, label: "\(q) recall", recall)
                g.computeMean(of: M.perQuestionHedging[q]!)
            }
        }
    }
}

/// MiniLM on CoreAI.framework, P(yes) = softmax(logits)[entailment] (calibrate.py's "raw, P=entail").
@available(macOS 27, *)
struct MiniLMRaw: TriageEvaluation {
    func probability(_ c: TriageCell) async throws -> Double { softmax(try await MiniLM.shared.logits(c), temperature: 1)[1] }
}

/// The same with the fitted temperature from models/decide/calibration.json (T = 4.38).
@available(macOS 27, *)
struct MiniLMTemperature: TriageEvaluation {
    static let temperature: Double = {
        struct Cal: Decodable { let temperature: Double }
        let data = try! Data(contentsOf: repo.appending(path: "models/decide/calibration.json"))
        return try! JSONDecoder().decode(Cal.self, from: data).temperature
    }()
    func probability(_ c: TriageCell) async throws -> Double {
        softmax(try await MiniLM.shared.logits(c), temperature: Self.temperature)[1]
    }
}

/// P = entail / (entail + contra), ignoring neutral.
@available(macOS 27, *)
struct MiniLMEntailVsContra: TriageEvaluation {
    func probability(_ c: TriageCell) async throws -> Double {
        let p = softmax(try await MiniLM.shared.logits(c), temperature: 1)
        return p[1] / (p[1] + p[0])
    }
}

/// Jev's recorded answers (`calibrate.py`'s first pass, 2026-09-19/20), replayed; no network.
@available(macOS 27, *)
struct JevReplay: TriageEvaluation {
    struct Missing: Error { let key: String }
    func probability(_ c: TriageCell) async throws -> Double {
        guard let p = Holdout.jev[c.jevKey]?[c.question] else { throw Missing(key: c.jevKey) }
        return p
    }
}

// MARK: - Tests: each run must reproduce the pooled row in docs/bench/decide-quality.md

struct Expected { let accuracy, ece, brier, hedging, auc: Double }

@available(macOS 27, *)
func check(_ e: Expected, tolerance: Double = 0.001) throws {
    let r = EvaluationContext.current.result
    print(r.groupedSummary)
    let got: [(String, Double, Double)] = [
        ("accuracy", r.aggregateValue(.mean(of: M.correct)), e.accuracy),
        ("ECE", r.aggregateValue(.custom(label: "ECE")), e.ece),
        ("Brier", r.aggregateValue(.mean(of: M.brier)), e.brier),
        ("hedging", r.aggregateValue(.mean(of: M.hedging)), e.hedging),
        ("AUC", r.aggregateValue(.custom(label: "AUC")), e.auc),
    ]
    for (name, value, want) in got {
        #expect(abs(value - want) <= tolerance, "\(name) \(value) vs calibrate.py \(want)")
    }
}

@Suite("Triage holdout on Evaluations", .serialized)
struct TriageHoldout {
    @available(macOS 27, *)
    @Test(.evaluates(MiniLMRaw(), info: ["subject": "MiniLM, CoreAI.framework, P=entail"]))
    func miniLMRaw() throws { try check(Expected(accuracy: 0.754, ece: 0.195, brier: 0.218, hedging: 0.452, auc: 0.712)) }

    @available(macOS 27, *)
    @Test(.evaluates(MiniLMTemperature(), info: ["subject": "MiniLM, CoreAI.framework, T=4.38, P=entail"]))
    func miniLMTemperature() throws { try check(Expected(accuracy: 0.723, ece: 0.058, brier: 0.185, hedging: 0.265, auc: 0.748)) }

    @available(macOS 27, *)
    @Test(.evaluates(MiniLMEntailVsContra(), info: ["subject": "MiniLM, CoreAI.framework, P=entail/(entail+contra)"]))
    func miniLMEntailVsContra() throws { try check(Expected(accuracy: 0.727, ece: 0.077, brier: 0.180, hedging: 0.390, auc: 0.804)) }

    @available(macOS 27, *)
    @Test(.evaluates(JevReplay(), info: ["subject": "Jev via Vercel AI Gateway, replayed from jev_cache.jsonl"]))
    func jev() throws { try check(Expected(accuracy: 0.898, ece: 0.039, brier: 0.070, hedging: 0.172, auc: 0.970)) }
}
#endif
