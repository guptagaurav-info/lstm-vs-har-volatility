"""Forecast losses, Diebold-Mariano test, Mincer-Zarnowitz, VaR coverage and a volatility-targeting backtest."""
import numpy as np
import pandas as pd
from scipy import stats


def qlike(true_var, fc_var):
    r = true_var / fc_var
    return r - np.log(r) - 1


def dm_test(loss_a, loss_b, lag):
    """Diebold-Mariano: mean of (loss_a - loss_b) with Newey-West variance. Negative t means model a is better."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d); m = d.mean(); u = d - m
    s = u @ u / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (u[k:] @ u[:-k]) / n
    t = m / np.sqrt(s / n)
    return float(t), float(2 * (1 - stats.norm.cdf(abs(t))))


def mincer_zarnowitz_r2(y, f):
    return float(np.corrcoef(y, f)[0, 1] ** 2)


def kupiec(breaches, n, p):
    x = breaches
    if x in (0, n):
        return float("nan")
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    return float(1 - stats.chi2.cdf(lr, 1))


def vol_target(ret_next: pd.Series, fc_ann_vol: pd.Series, target, max_lev, every, cost_bps):
    """Hold SPY at weight target / forecast vol (capped), rebalanced every `every` days; returns daily P&L series."""
    w = (target / fc_ann_vol).clip(upper=max_lev)
    w_held = w.copy()
    w_held[:] = np.nan
    w_held.iloc[::every] = w.iloc[::every]
    w_held = w_held.ffill()
    turn = w_held.diff().abs().fillna(w_held.iloc[0])
    pnl = w_held * ret_next - turn * cost_bps / 1e4
    return pnl, w_held


def perf(pnl: pd.Series):
    ann = pnl.mean() * 252; vol = pnl.std() * np.sqrt(252)
    eq = np.exp(pnl.cumsum()); dd = (eq / eq.cummax() - 1).min()
    return {"ann_return": float(ann), "ann_vol": float(vol), "sharpe": float(ann / vol), "max_dd": float(dd)}
