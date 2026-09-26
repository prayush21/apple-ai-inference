We're refining a product idea before building anything. This session is **brainstorming and interviewing only: no code, no files except the final write-up.** Interview me until the idea is sharp enough to spec. Use the `grill-me` skill if it's available.

## The idea: "Pull"

Social feeds, subscriptions and notifications pull my attention toward goals that aren't mine, such as someone else's screen-time targets. Pull reverses that: my own goals, habits, systems and projects get an equally strong pull, bringing me back to my work when I drift. Think of it as a personal user-research scientist that watches how I work and acts for *my* best outcome, not a platform's.

It barely needs its own interface. It lives in the OS: notifications, a menu-bar presence, cards that appear when needed. The experience itself is the intelligence. It decides *whether*, *when* and *how* to interrupt, how to phrase the nudge, and what small UI to show. It isn't a chat window.

## Working architecture (a hypothesis to challenge, not a decision)

1. **Decide/choose (fast, on-device, "system 1"):** runs on every context change in ~25 ms. Am I drifting or on-track (and what kind: feed spiral, fatigue, legitimate break)? Is now a good moment to interrupt? Which goal or project to pull me toward? How firmly: do nothing, badge, notification, full-screen card, or later?
2. **Compose (small on-device LLM, only when acting):** writes the nudge, specific and pointing at my real next step, plus a UI description in a fixed format that plain SwiftUI renders. The model never owns app state.
3. **Learn (on-device):** did I come back, dismiss it, or snooze it? Each reaction becomes a label, so it adapts to me without my behaviour leaving the device.

Current thinking: **start on macOS, not iOS.**
- iOS's Screen Time extensions have tight memory limits (probably far too small for a ~300 MB model; not yet verified), and other apps' content isn't visible.
- On macOS, with permission, a menu-bar app can see the frontmost app, window titles, idle time and my calendar.
- My projects live on the Mac (git repos, TODOs), so nudges can name a concrete next step instead of a generic "focus!".

Proposed first milestone: **observe before intervening.** Log context snapshots locally for 3–5 days (no nudges), I write down my goals, then I label ~150 snapshots to form the benchmark.

## What already exists (repo `apple-ai-inference`, branch `system-one`)

- **On-device model:** MiniLM (82M, NLI cross-encoder) converted to a Core AI `.aimodel` and running in-process in Swift on `CoreAI.framework` (macOS 27 / Xcode 27). **5.3 ms for one question, 25 ms for a 5-question request**, 41–91× faster than the old runtime and ~10× faster than Jev.
- **Quality on our 150-message triage holdout:**
  - Jev (TypeSafe, cloud): 0.898 accuracy, AUC 0.97.
  - Laya (421M, open, PyTorch only): 0.824 accuracy, AUC 0.86.
  - MiniLM zero-shot: 0.754 accuracy, AUC 0.71, and it never says yes on 3 of the 5 questions.
  So on-device speed is solved; quality is the gap. The plan to close it is distilling Jev's probabilities into MiniLM.
- **Other candidates:** OpenJev (`heman10x/rlcd-modernbert-151m`, a pick-one-of-N model with an abstain option; not tested yet) and SmolLM2-360M for the compose layer (scaffolded on branch `llm`).
- **Evaluation harness:** Xcode 27's Evaluations framework, proven on the triage holdout (reproduces the Python numbers within 0.001; notes in `docs/bench/eval/README.md`). It can test any model, and it has synthetic-sample generation, LLM-as-judge and tool-call checks.
- Read `README.md`, `docs/bench/README.md` and `docs/bench/eval/README.md` only if you need specifics. Don't re-derive the findings.

## Interview me on (one topic at a time, 1–3 questions per turn)

1. **The core problem:** what exactly pulls me away today, when, and on which device? What does a "good day" look like? What have I tried (Screen Time, blockers, Pomodoro, todo apps), and why did it fail?
2. **Goals model:** what's a goal vs a project vs a habit vs a system? How many at once? Where do they live today (repos, notes, calendar)? How does Pull learn my *next step*, and who keeps it current?
3. **Drift:** what counts as drifting vs resting vs legitimate research? Is YouTube drift? Is Slack? How would I label a snapshot?
4. **Interventions:** which forms (badge, notification, card, blocking, a question back to me) and how strong can it get? Can it be stubborn? What must it never do (interrupt meetings, shame me)? What's the budget of nudges per day before it becomes noise?
5. **Timing:** what makes a moment interruptible? Is a reminder after the fact ("you spent 40 min on X") ever useful, or only in-the-moment pulls?
6. **Framing and tone:** coach, mirror, friend, drill sergeant? Does it ask questions ("what's blocking you?") or state facts? Should it adapt its tone to me?
7. **Privacy and trust:** what may be logged (app names, window titles, URLs, screenshots?), for how long, and where? What would make me turn it off?
8. **Success metrics:** how do we know it works: time back on goal, fewer drift minutes, my own daily rating? What's the benchmark's ground truth?
9. **Scope of v0:** the smallest version I'd actually use tomorrow. Mac only? One goal? Rules first, models later?
10. **Stress-test the architecture:** does each decision really need a model, or would a rule do? Where does on-device genuinely matter vs just being nice? Is the three-layer split right?

Along the way, **brainstorm**: propose alternatives and bolder variants, and name risks I haven't raised (nag fatigue, gaming the metric, lying to it, dependency). Push back when my answers are vague or contradict each other.

## Output when we're done (only after I say we're done)

Write `docs/pull/idea.md`:
- one-paragraph pitch;
- the user (me) and the problem, in my words;
- goals model;
- decision catalog: each decision's inputs, output type (decide / choose / score / compose), what a good answer is, and the cost of a wrong one;
- intervention ladder and hard rules;
- context-snapshot fields and privacy rules;
- success metrics;
- v0 scope and the first milestone;
- open questions and risks.

Keep it short enough to read in 5 minutes.
