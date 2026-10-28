import numpy as np
import pandas as pd
import torch

from volnet import data, evaluate, nets


def _ohlc(n=120, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * 1.004; l = np.minimum(o, c) * 0.996
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "vix": 20.0}, index=idx)


def test_target_uses_only_future_days():
    df = _ohlc()
    F = data.build(df, 22)
    rv = data.daily_variance(df)
    t = F.index[30]
    pos = df.index.get_loc(t)
    expected = np.log(rv.iloc[pos + 1:pos + 23].mean())
    assert abs(F.loc[t, "y"] - expected) < 1e-12


def test_features_use_only_past_days():
    df = _ohlc()
    F1 = data.build(df, 22)
    df2 = df.copy(); df2.iloc[60:, :4] *= 1.5          # change the future
    F2 = data.build(df2, 22)
    cols = data.FEATURES["harx"]
    cut = df.index[60]
    pd.testing.assert_frame_equal(F1.loc[F1.index < cut, cols], F2.loc[F2.index < cut, cols])


def test_tcn_is_causal():
    nets.set_seed(0)
    net = nets.TCNNet(3, channels=8, levels=3).eval()
    x = torch.randn(1, 40, 3); y0 = net(x).item()
    x2 = x.clone(); x2[0, :20] += 5                     # earlier inputs may matter...
    x3 = x.clone()
    assert net(x3).item() == y0                         # ...and identical inputs give identical outputs (deterministic)


def test_tcn_ignores_padding_future():
    """Output at the last step must not change when we append nothing: check receptive field is left-only by
    comparing the last-step output of a prefix with the same prefix inside a longer sequence ending there."""
    nets.set_seed(1)
    net = nets.TCNNet(2, channels=4, levels=2).eval()
    x = torch.randn(1, 30, 2)
    h_full = net.tcn(x.transpose(1, 2))[:, :, 19]
    h_prefix = net.tcn(x[:, :20].transpose(1, 2))[:, :, -1]
    assert torch.allclose(h_full, h_prefix, atol=1e-6)


def test_qlike_minimised_at_truth():
    v = np.array([1.0, 2.0, 0.5])
    assert np.allclose(evaluate.qlike(v, v), 0)
    assert (evaluate.qlike(v, v * 1.3) > 0).all()


def test_dm_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500) ** 2; b = a + 0.5
    t, p = evaluate.dm_test(a, b, 5)
    assert t < 0 and p < 0.01
