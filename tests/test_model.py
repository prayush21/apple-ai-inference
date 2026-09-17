import torch

from snake_ai.model import SnakeModelConfig, SnakeTransformer, SnakeTransformerStateful

CFG = SnakeModelConfig(d_model=32, n_heads=2, n_layers=2, max_seq_len=32)


def _pair():
    torch.manual_seed(0)
    stateless = SnakeTransformer(CFG).eval()
    stateful = SnakeTransformerStateful(CFG).eval()
    stateful.load_state_dict(stateless.state_dict(), strict=False)
    return stateless, stateful


def test_shapes():
    m = SnakeTransformer(CFG)
    assert m(torch.randn(2, 7, 16)).shape == (2, 7, 4)


def test_stateful_matches_stateless_token_by_token():
    stateless, stateful = _pair()
    T = 10
    feats = torch.randn(1, T, 16)
    with torch.no_grad():
        ref = stateless(feats)
        outs = []
        for t in range(T):
            out = stateful(feats[:, t : t + 1], torch.tensor([[t]]))
            outs.append(out)
        got = torch.cat(outs, dim=1)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def test_stateful_prefill_then_decode():
    """Multi-token prefill followed by single-token steps must match too."""
    stateless, stateful = _pair()
    T = 8
    feats = torch.randn(1, T, 16)
    with torch.no_grad():
        ref = stateless(feats)
        prefill = stateful(feats[:, :5], torch.arange(5)[None])
        rest = [stateful(feats[:, t : t + 1], torch.tensor([[t]])) for t in range(5, T)]
        got = torch.cat([prefill, *rest], dim=1)
    assert torch.allclose(ref, got, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    m = SnakeTransformer(CFG)
    m.save_checkpoint(tmp_path / "m.pt")
    m2 = SnakeTransformerStateful.load_checkpoint(tmp_path / "m.pt")
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        assert torch.allclose(m(x)[:, -1], m2(x, torch.arange(3)[None])[:, -1], atol=1e-5)
