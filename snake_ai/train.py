"""Train ``SnakeTransformer`` by imitating the heuristic policy.

    python -m snake_ai.train --episodes 2000 --epochs 8 --out checkpoints/snake.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_dataset
from .model import SnakeModelConfig, SnakeTransformer


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
) -> SnakeTransformer:
    torch.manual_seed(seed)
    cfg = cfg or SnakeModelConfig()
    assert seq_len <= cfg.max_seq_len

    t0 = time.time()
    X, Y, M = build_dataset(episodes, seq_len=seq_len, seed=seed)
    print(f"dataset: {X.shape[0]} windows x {seq_len} steps ({M.sum():.0f} labelled steps) in {time.time() - t0:.1f}s")

    # Hold out 10% of windows for validation.
    n_val = max(1, len(X) // 10)
    perm = np.random.default_rng(seed).permutation(len(X))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X, Y, M = (torch.from_numpy(a) for a in (X, Y, M))
    Xv, Yv, Mv = X[val_idx], Y[val_idx], M[val_idx]
    Xt, Yt, Mt = X[tr_idx], Y[tr_idx], M[tr_idx]

    model = SnakeTransformer(cfg)
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
            f"epoch {epoch:2d}  train loss {tot_loss / n_batches:.3f} acc {tot_acc / n_batches:.3f}"
            f"  |  val loss {vloss:.3f} acc {vacc:.3f}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(out)
    print(f"saved {out}")
    return model


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/snake.pt"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    train(
        episodes=a.episodes,
        epochs=a.epochs,
        seq_len=a.seq_len,
        batch_size=a.batch_size,
        lr=a.lr,
        out=a.out,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
