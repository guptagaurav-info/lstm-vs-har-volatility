"""Charts for the write-up, drawn from the walk-forward predictions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

NAVY, GOLD, GREY, RED, TEAL, PURPLE = "#1f3a5f", "#c9a227", "#8a8f98", "#b23a3a", "#2a7f7f", "#6b4c9a"
COL = {"naive": GREY, "vix": "#9db4cf", "har": "#5d82ad", "harx": NAVY, "lgbm": TEAL, "lstm": RED, "tcn": GOLD}
NAME = {"naive": "Last month", "vix": "VIX only", "har": "HAR", "harx": "HAR + VIX", "lgbm": "Gradient boosting", "lstm": "LSTM", "tcn": "TCN"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12.5, "figure.dpi": 150, "savefig.bbox": "tight"})


def ann_vol(pred, k):
    return np.sqrt(252 * np.exp(pred[k] + 0.5 * pred[f"s2_{k}"])) * 100


def forecast_path(pred, path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    real = np.sqrt(252 * np.exp(pred["y"])) * 100
    wk = pred.index[::5]
    ax.plot(wk, real.loc[wk], color="black", lw=1.1, label="Realised over the next month")
    for k in ["harx", "lstm"]:
        ax.plot(wk, ann_vol(pred, k).loc[wk], color=COL[k], lw=1.3, alpha=0.9, label=f"Forecast: {NAME[k]}")
    ax.set_yscale("log"); ax.set_yticks([10, 20, 40, 80]); ax.set_yticklabels(["10%", "20%", "40%", "80%"])
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Annualised volatility (log scale)")
    ax.set_title("Next-month S&P 500 volatility: forecast vs what happened, 2008–2026\nEach model refit every January on data before it")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.savefig(path); plt.close(fig)


def loss_bars(res, path):
    ks = sorted(res, key=lambda k: res[k]["qlike"])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    v = [res[k]["qlike"] for k in ks]
    b = ax.barh([NAME[k] for k in ks][::-1], v[::-1], color=[COL[k] for k in ks][::-1])
    for r, k in zip(b, ks[::-1]):
        t = res[k].get("dm_vs_harx_t")
        lab = f"{res[k]['qlike']:.3f}" + ("" if t is None else f"   DM t vs HAR+VIX {t:+.1f}")
        ax.text(r.get_width() + 0.002, r.get_y() + r.get_height() / 2, lab, va="center", fontsize=9)
    ax.set_xlabel("Average forecast loss (QLIKE; lower is better)")
    ax.set_xlim(0, max(v) * 1.45)
    ax.set_title("Out-of-sample forecast loss, January 2008 to August 2026 (4,688 days)")
    fig.savefig(path); plt.close(fig)


def by_year(by, path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    rel = by.div(by["harx"], axis=0) - 1
    for k in ["har", "lgbm", "lstm", "tcn"]:
        ax.plot(rel.index, rel[k] * 100, marker="o", lw=1.8 if k in ("lstm", "tcn") else 1.2, color=COL[k], label=NAME[k])
    ax.axhline(0, color=NAVY, lw=1.2, ls="--", label="HAR + VIX (baseline)")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylabel("Loss vs HAR + VIX (%; below 0 = better)")
    ax.set_title("Year by year, no model beat HAR + VIX consistently")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.savefig(path); plt.close(fig)


def covid(pred, path):
    s = pred.loc["2020-01-15":"2020-06-30"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    real = np.sqrt(252 * np.exp(s["y"])) * 100
    ax.plot(s.index, real, color="black", lw=2, label="Realised over the next month")
    for k in ["vix", "harx", "lstm", "tcn"]:
        ax.plot(s.index, ann_vol(s, k), color=COL[k], lw=1.5, label=NAME[k])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("March 2020: every forecast, the VIX included, was weeks late")
    ax.legend(frameon=False, fontsize=9.5)
    fig.savefig(path); plt.close(fig)


def vol_target_plot(daily, target, path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k in ["naive", "harx", "lstm"]:
        rv = daily[k].rolling(252).std() * np.sqrt(252) * 100
        ax.plot(rv.index, rv, color=COL[k], lw=1.5, label=f"Sized with {NAME[k]}")
    ax.axhline(target * 100, color="black", ls="--", lw=1, label="15% target")
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Trailing 1-year realised volatility (%)")
    ax.set_title("Volatility-targeted S&P 500: how close each forecast kept the book to 15%")
    ax.legend(frameon=False, fontsize=9.5)
    fig.savefig(path); plt.close(fig)


def make_all(out, F, pred, by, biz, daily, cfg):
    f = out / "figures"
    forecast_path(pred, f / "forecast_path.png")
    import json
    res = json.load(open(out / "metrics.json"))["forecast"]
    loss_bars(res, f / "loss_bars.png")
    by_year(by, f / "loss_by_year.png")
    covid(pred, f / "covid_2020.png")
    vol_target_plot(daily, cfg["vol_target"]["target"], f / "vol_target.png")
