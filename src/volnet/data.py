"""Load SPY daily OHLC and the CBOE VIX, build the daily variance proxy, features and the one-month target."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load(cfg) -> pd.DataFrame:
    t = pq.read_table(cfg["prices"], filters=[("symbol", "=", cfg["symbol"])]).to_pandas()
    t["date"] = pd.to_datetime(t["report_date"])
    for c in ["open", "high", "low", "close"]:
        t[c] = t[c].astype(float)
    t = t.sort_values("date").set_index("date")[["open", "high", "low", "close"]]
    v = pd.read_csv(cfg["vix"], parse_dates=["DATE"]).rename(columns={"DATE": "date", "CLOSE": "vix"}).set_index("date")[["vix"]]
    df = t.join(v, how="inner")
    df = df[(df.index >= cfg["start"]) & (df.index <= cfg["end"])]
    # data checks: positive prices, high >= low, no absurd daily moves (unadjusted split would show as one)
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df["low"]).all()
    assert (np.abs(np.log(df["close"]).diff().dropna()) < 0.25).all()
    return df


def daily_variance(df: pd.DataFrame) -> pd.Series:
    """Overnight squared return plus the Garman-Klass open-to-close range estimator (daily variance proxy)."""
    o, h, l, c = (np.log(df[k]) for k in ["open", "high", "low", "close"])
    overnight = (o - c.shift(1)) ** 2
    gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    rv = overnight + gk.clip(lower=0)
    return rv.clip(lower=1e-8)


def build(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret"] = np.log(df["close"]).diff()
    out["rv"] = daily_variance(df)
    out["vixvar"] = (df["vix"] / 100) ** 2 / 252          # VIX converted to a daily variance
    out["log_rv_d"] = np.log(out["rv"])
    out["log_rv_w"] = np.log(out["rv"].rolling(5).mean())
    out["log_rv_m"] = np.log(out["rv"].rolling(22).mean())
    out["log_vix"] = np.log(out["vixvar"])
    out["neg_ret_d"] = np.minimum(out["ret"], 0)
    out["neg_ret_w"] = np.minimum(out["ret"].rolling(5).sum(), 0)
    # target: log of the average daily variance over the NEXT `horizon` days (t+1 .. t+h); known only at t+h
    fwd = out["rv"][::-1].rolling(horizon).mean()[::-1].shift(-1)
    out["y"] = np.log(fwd)
    out["ret_next"] = out["ret"].shift(-1)
    return out.dropna(subset=["ret", "log_rv_m", "log_vix"])


FEATURES = {
    "har": ["log_rv_d", "log_rv_w", "log_rv_m"],
    "harx": ["log_rv_d", "log_rv_w", "log_rv_m", "log_vix", "neg_ret_d", "neg_ret_w"],
    "vix": ["log_vix"],
}
SEQ_COLS = ["log_rv_d", "log_vix", "ret"]
