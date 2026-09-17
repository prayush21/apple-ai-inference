"""Train ``SnakeTransformer`` by imitating the heuristic policy.

Stage 1 is plain behaviour cloning on teacher rollouts. Optional DAgger rounds
then roll out the *learner*, label what it sees with the teacher, aggregate,
and keep training — this is what turns "imitates the teacher 66% of the time"
into "actually survives".

    python -m snake_ai.train --episodes 4000 --epochs 10 --dagger-rounds 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Episode, make_teacher, pack_windows, simulate_dagger_episode, simulate_episode
from .evaluate import choose_with_model, evaluate
from .model import SnakeModelConfig, SnakeTransformer


def _split(X, Y, M, seed: int, val_frac: float = 0.1):
    n_val = max(1, int(len(X) * val_frac))
    perm = np.random.default_rng(seed).permutation(len(X))
    t = lambda a: torch.from_numpy(a)
    v, tr = perm[:n_val], perm[n_val:]
    return (t(X[tr]), t(Y[tr]), t(M[tr])), (t(X[v]), t(Y[v]), t(M[v]))


def fit(model: SnakeTransformer, episodes: list[Episode], *, seq_len: int, epochs: int, batch_size: int, lr: float, seed: int, tag: str = "") -> None:
    """Train ``model`` in place on ``episodes`` for ``epochs``."""
    cfg = model.cfg
    X, Y, M = pack_windows(episodes, seq_len)
    (Xt, Yt, Mt), (Xv, Yv, Mv) = _split(X, Y, M, seed)
    print(f"{tag}dataset: {len(episodes)} episodes -> {len(X)} windows x {seq_len} ({int(M.sum())} labelled steps)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def masked_loss_acc(logits, y, m):
        loss = F.cross_entropy(logits.reshape(-1, cfg.n_actions), y.reshape(-1), reduction="none")
        loss = (loss * m.reshape(-1)).sum() / m.sum()
        correct = ((logits.argmax(-1) == y).float() * m).sum() / m.sum()
        return loss, correct

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xt))
        tot_loss = tot_acc = 0.0
        n_batches = 0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            loss, acc = masked_loss_acc(model(Xt[idx]), Yt[idx], Mt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            tot_acc += acc.item()
            n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vloss, vacc = masked_loss_acc(model(Xv), Yv, Mv)
        print(
            f"{tag}epoch {epoch:2d}  train loss {tot_loss / n_batches:.3f} acc {tot_acc / n_batches:.3f}"
            f"  |  val loss {vloss:.3f} acc {vacc:.3f}"
        )


def model_actor(model: SnakeTransformer):
    """Wrap the model as a DAgger actor (uses the same safe-argmax as play)."""
    def act(game, history):
        return choose_with_model(model, history[-model.cfg.max_seq_len :], game, 0, safe_only=True)
    return act


def train(
    *,
    episodes: int,
    epochs: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    out: Path,
    seed: int = 0,
    cfg: SnakeModelConfig | None = None,
    eval_games: int = 30,
    dagger_rounds: int = 0,
    dagger_games: int = 300,
    dagger_epochs: int = 4,
    teacher: str = "heuristic",
    teacher_depth: int = 1,
) -> SnakeTransformer:
    torch.manual_seed(seed)
    teacher_policy = make_teacher(teacher, teacher_depth)
    cfg = cfg or SnakeModelConfig()
    assert seq_len <= cfg.max_seq_len
    model = SnakeTransformer(cfg)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params  {cfg}")

    # Stage 1: behaviour cloning on teacher rollouts.
    t0 = time.time()
    data: list[Episode] = [simulate_episode(seed + ep, teacher=teacher_policy) for ep in range(episodes)]
    print(f"simulated {episodes} episodes with teacher={teacher} in {time.time() - t0:.1f}s")
    fit(model, data, seq_len=seq_len, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)
    if eval_games:
        print("after cloning, vs heuristic:", evaluate(model, games=eval_games))

    # Stage 2: DAgger — aggregate learner-visited states with teacher labels.
    next_seed = seed + episodes
    for r in range(1, dagger_rounds + 1):
        t0 = time.time()
        actor = model_actor(model)
        new = [simulate_dagger_episode(next_seed + i, actor, teacher=teacher_policy) for i in range(dagger_games)]
        next_seed += dagger_games
        data.extend(new)
        print(f"[dagger {r}] rolled out {dagger_games} learner episodes "
              f"(avg {np.mean([len(a) for _, a in new]):.0f} steps) in {time.time() - t0:.1f}s")
        fit(model, data, seq_len=seq_len, epochs=dagger_epochs, batch_size=batch_size, lr=lr * 0.5,
            seed=seed + r, tag=f"[dagger {r}] ")
        if eval_games:
            print(f"[dagger {r}] vs heuristic:", evaluate(model, games=eval_games))

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(out)
    print(f"saved {out}")
    return model


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/snake.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--eval-games", type=int, default=30)
    ap.add_argument("--dagger-rounds", type=int, default=0)
    ap.add_argument("--dagger-games", type=int, default=300)
    ap.add_argument("--dagger-epochs", type=int, default=4)
    ap.add_argument("--teacher", choices=["heuristic", "minimax"], default="heuristic")
    ap.add_argument("--teacher-depth", type=int, default=1, help="minimax lookahead in own moves")
    a = ap.parse_args(argv)
    train(
        cfg=SnakeModelConfig(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads),
        eval_games=a.eval_games,
        episodes=a.episodes,
        epochs=a.epochs,
        seq_len=a.seq_len,
        batch_size=a.batch_size,
        lr=a.lr,
        out=a.out,
        seed=a.seed,
        dagger_rounds=a.dagger_rounds,
        dagger_games=a.dagger_games,
        dagger_epochs=a.dagger_epochs,
        teacher=a.teacher,
        teacher_depth=a.teacher_depth,
    )


if __name__ == "__main__":
    main()
