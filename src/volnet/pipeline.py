"""Walk-forward test: can an LSTM or TCN beat HAR (and the VIX) at forecasting next-month S&P 500 volatility?

Each January from 2008 every model is refit on all data before that year (minus a 22-day embargo), then
forecasts every day of the year. Sequence models are averaged over three seeds. Predictions are cached
per year so the run can resume.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from volnet import data, evaluate, nets

MODELS = ["naive", "vix", "har", "harx", "lgbm", "lstm", "tcn"]
LABEL = {"naive": "Last month's realised (random walk)", "vix": "VIX only (implied)", "har": "HAR (daily/weekly/monthly)",
         "harx": "HAR + VIX + leverage", "lgbm": "Gradient boosting (HAR-X inputs)", "lstm": "LSTM (60-day sequences)",
         "tcn": "TCN (60-day sequences)"}


def ols(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); b = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return A @ b, np.column_stack([np.ones(len(Xte)), Xte]) @ b


def sequences(F: pd.DataFrame, cols, L, idx):
    """Windows of the last L days (ending at each index position) for the given row positions."""
    M = F[cols].to_numpy(np.float32)
    return np.stack([M[i - L + 1:i + 1] for i in idx])


def fit_year(F, year, cfg):
    L, h = cfg["seq_len"], cfg["horizon"]
    dates = F.index
    pos = np.arange(len(F))
    tr = pos[(dates < f"{year}-01-01") & F["y"].notna().to_numpy() & (pos >= L)]
    tr = tr[:-cfg["embargo"]]                                   # embargo: no training target may reach into the test year
    te = pos[(dates.year == year) & F["y"].notna().to_numpy()]
    assert dates[tr].max() < dates[te].min()
    # the last training target window must end before the test year starts
    assert tr.max() + h < te.min()
    ytr, yte = F["y"].to_numpy()[tr], F["y"].to_numpy()[te]
    P, S2 = {}, {}

    P["naive"] = F["log_rv_m"].to_numpy()[te]; S2["naive"] = np.var(ytr - F["log_rv_m"].to_numpy()[tr])
    for k in ["vix", "har", "harx"]:
        cols = data.FEATURES[k]
        fit, pr = ols(F[cols].to_numpy()[tr], ytr, F[cols].to_numpy()[te])
        P[k], S2[k] = pr, np.var(ytr - fit)

    import lightgbm as lgb
    p = dict(cfg["lgbm"]); n = p.pop("n_estimators")
    m = lgb.train(dict(objective="regression", seed=cfg["seed"], deterministic=True, force_row_wise=True, num_threads=2, verbose=-1, **p),
                  lgb.Dataset(F[data.FEATURES["harx"]].to_numpy()[tr], ytr), n)
    P["lgbm"] = m.predict(F[data.FEATURES["harx"]].to_numpy()[te])
    S2["lgbm"] = np.var(ytr - m.predict(F[data.FEATURES["harx"]].to_numpy()[tr]))

    # sequence models: standardise inputs and target with training statistics only
    mu, sd = F[data.SEQ_COLS].iloc[tr].mean(), F[data.SEQ_COLS].iloc[tr].std()
    Z = (F[data.SEQ_COLS] - mu) / sd
    ym, ys = ytr.mean(), ytr.std()
    Xtr, Xte = sequences(Z, data.SEQ_COLS, L, tr), sequences(Z, data.SEQ_COLS, L, te)
    for name, Net, hp in [("lstm", nets.LSTMNet, cfg["lstm"]), ("tcn", nets.TCNNet, cfg["tcn"])]:
        preds, fits = [], []
        for s in cfg["seeds_nn"]:
            nets.set_seed(s)
            kw = {k: v for k, v in hp.items() if k not in ("epochs", "lr", "batch", "weight_decay")}
            net = Net(len(data.SEQ_COLS), **kw)
            nets.train(net, Xtr, (ytr - ym) / ys, hp["epochs"], hp["lr"], hp["batch"], hp["weight_decay"], s)
            preds.append(nets.predict(net, Xte) * ys + ym); fits.append(nets.predict(net, Xtr) * ys + ym)
        P[name] = np.mean(preds, axis=0); S2[name] = np.var(ytr - np.mean(fits, axis=0))
    out = pd.DataFrame({k: P[k] for k in MODELS}, index=dates[te])
    out["y"] = yte
    for k in MODELS:
        out[f"s2_{k}"] = S2[k]
    return out, len(tr)


def run_walkforward(F, cfg, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    parts, log = [], {}
    for year in range(cfg["first_test_year"], F.index.year.max() + 1):
        f = cache / f"pred_{year}.pkl"
        if f.exists():
            parts.append(pd.read_pickle(f)); continue
        t0 = time.time()
        out, ntr = fit_year(F, year, cfg)
        out.to_pickle(f); parts.append(out)
        print(f"{year}: train {ntr} days, test {len(out)}, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(parts)


def score(pred: pd.DataFrame, F: pd.DataFrame, cfg):
    yv = np.exp(pred["y"])                                        # realised next-month average daily variance
    res, losses = {}, {}
    for k in MODELS:
        fv = np.exp(pred[k] + 0.5 * pred[f"s2_{k}"])                # log-normal bias correction from training residuals
        ql = evaluate.qlike(yv, fv); losses[k] = ql
        res[k] = {"label": LABEL[k], "mse_log": float(np.mean((pred["y"] - pred[k]) ** 2)), "qlike": float(ql.mean()),
                  "mz_r2": evaluate.mincer_zarnowitz_r2(pred["y"], pred[k]),
                  "mae_vol_pts": float(np.mean(np.abs(np.sqrt(252 * yv) - np.sqrt(252 * fv))) * 100)}
    for k in MODELS:
        if k != "harx":
            t, p = evaluate.dm_test(losses[k], losses["harx"], cfg["horizon"])
            res[k]["dm_vs_harx_t"], res[k]["dm_vs_harx_p"] = t, p
    # by year (QLIKE)
    by_year = pd.DataFrame({k: losses[k] for k in MODELS}).groupby(pred.index.year).mean()
    return res, by_year, losses


def business(pred, F, cfg):
    vt = cfg["vol_target"]; out = {}
    ret_next = F.loc[pred.index, "ret_next"].fillna(0)
    daily = {}
    bh = ret_next.copy()
    out["buy_and_hold"] = {**evaluate.perf(bh), "avg_abs_miss_vs_target_pts": float(np.mean(np.abs(bh.groupby(bh.index.year).std() * np.sqrt(252) - vt["target"])) * 100)}
    for k in MODELS:
        fc = np.sqrt(252 * np.exp(pred[k] + 0.5 * pred[f"s2_{k}"]))
        pnl, w = evaluate.vol_target(ret_next, fc, vt["target"], vt["max_leverage"], vt["rebalance_days"], vt["cost_bps"])
        yr_vol = pnl.groupby(pnl.index.year).std() * np.sqrt(252)
        out[k] = {**evaluate.perf(pnl), "avg_abs_miss_vs_target_pts": float(np.mean(np.abs(yr_vol - vt["target"])) * 100),
                  "worst_year_vol": float(yr_vol.max()), "avg_turnover_yr": float(w.diff().abs().sum() / (len(w) / 252))}
        daily[k] = pnl
        # 1-day 99% VaR coverage
        if True:
            s1 = np.sqrt(np.exp(pred[k] + 0.5 * pred[f"s2_{k}"]))
            var = 2.326 * s1
            br = int((ret_next < -var).sum()); n = int(len(var))
            out[k].update({"var99_breaches": br, "var99_rate": br / n, "var99_kupiec_p": evaluate.kupiec(br, n, 1 - cfg["var_level"])})
    return out, pd.DataFrame(daily)


def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["out_dir"]); (out / "figures").mkdir(parents=True, exist_ok=True)
    raw = data.load(cfg)
    F = data.build(raw, cfg["horizon"])
    pred = run_walkforward(F, cfg, Path("data/cache"))
    res, by_year, losses = score(pred, F, cfg)
    biz, daily = business(pred, F, cfg)
    pred.to_csv(out / "predictions.csv"); by_year.to_csv(out / "qlike_by_year.csv")
    metrics = {"sample": {"days": int(len(F)), "first": str(F.index.min().date()), "last": str(F.index.max().date()),
                          "test_days": int(len(pred)), "test_first": str(pred.index.min().date()), "test_last": str(pred.index.max().date())},
               "forecast": res, "vol_target": biz}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, default=float)
    from volnet import figures
    figures.make_all(out, F, pred, by_year, biz, daily, cfg)
    print(json.dumps(metrics, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:2])
