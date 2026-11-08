"""
Shared strategies for backtest and live alerts.

EMA modes ("1h", "4h"): trend-pullback with EMA/RSI/ADX, ATR-based stop and target.

BOS + POC modes ("poc1h", "poc4h"): market-structure strategy
  1. Swing highs/lows are detected with pivots (confirmed only AFTER `pivot` bars, no look-ahead).
  2. BOS (break of structure): a candle CLOSES beyond the last swing high -> bullish bias,
     or below the last swing low -> bearish bias.
  3. The higher timeframe (4H for 1H entries, Daily for 4H entries) must show the same bias.
  4. POC (point of control): the price level where price spent the most time over the last N candles
     (time-based, TPO style, because free forex data has no real volume).
  5. Entry: after a bullish BOS, price pulls back to the POC and a bullish candle closes back above it
     (bearish setup is the mirror image). Structure must still be intact.
  6. Stop: beyond the last 5 candles' extreme (1 to 3 ATR away). Target: 2R.

Higher-timeframe values are shifted by one bar so only COMPLETED bars are used.
"""
import numpy as np
import pandas as pd

_NAMES = """EURUSD GBPUSD AUDUSD NZDUSD USDJPY USDCHF USDCAD
EURGBP EURJPY EURCHF EURAUD EURCAD EURNZD
GBPJPY GBPCHF GBPAUD GBPCAD GBPNZD
AUDJPY AUDCHF AUDCAD AUDNZD
NZDJPY NZDCHF NZDCAD
CADJPY CADCHF CHFJPY""".split()
PAIRS = {p: f"{p}=X" for p in _NAMES}  # all 28 major + cross pairs

SL_ATR = 1.5
TP_ATR = 3.0
POC_RR = 2.0     # reward:risk for BOS+POC modes
POC_TOL = 0.25   # how far (in ATR) below/above the POC a pullback may stop and still count

MODES = {
    "1h": {"kind": "ema", "entry": "1h", "trend": "4h", "fast": 50, "slow": 200, "adx_min": 20,
           "max_hold": 48, "delta": "1h", "label": "1H EMA"},
    "4h": {"kind": "ema", "entry": "4h", "trend": "1D", "fast": 20, "slow": 50, "adx_min": 20,
           "max_hold": 30, "delta": "4h", "label": "4H EMA"},
    "poc1h": {"kind": "poc", "entry": "1h", "trend": "4h", "pivot": 3, "poc_window": 100,
              "bos_window": 40, "max_hold": 48, "delta": "1h", "label": "1H BOS+POC"},
    "poc4h": {"kind": "poc", "entry": "4h", "trend": "1D", "pivot": 3, "poc_window": 60,
              "bos_window": 20, "max_hold": 30, "delta": "4h", "label": "4H BOS+POC"},
}
GROUPS = {"both": ["1h", "4h"], "ema": ["1h", "4h"], "poc": ["poc1h", "poc4h"],
          "all": ["1h", "4h", "poc1h", "poc4h"]}


def resolve_modes(name):
    return GROUPS.get(name, [name])


# ---------- indicators ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


# ---------- market structure: swings, BOS, bias ----------
def structure(df, k=3):
    """
    Returns a DataFrame with, for every candle:
      bias     1 bullish / -1 bearish / 0 unknown (direction of the latest BOS)
      since    candles since the latest BOS
      last_sh  most recent CONFIRMED swing high level
      last_sl  most recent CONFIRMED swing low level
    A pivot at candle j is only known at candle j+k, so there is no look-ahead.
    """
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    ph = (df["High"].rolling(2 * k + 1, center=True).max() == df["High"]).values
    pl = (df["Low"].rolling(2 * k + 1, center=True).min() == df["Low"]).values

    bias = np.zeros(n, dtype=int)
    since = np.full(n, 10**6, dtype=int)
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    sh = sl = np.nan
    sh_open = sl_open = False
    b, s = 0, 10**6
    for t in range(n):
        j = t - k
        if j >= 0 and ph[j]:
            sh, sh_open = H[j], True
        if j >= 0 and pl[j]:
            sl, sl_open = L[j], True
        if sh_open and C[t] > sh:      # bullish BOS: close above last swing high
            b, s, sh_open = 1, 0, False
        elif sl_open and C[t] < sl:    # bearish BOS: close below last swing low
            b, s, sl_open = -1, 0, False
        else:
            s = min(s + 1, 10**6)
        bias[t], since[t], last_sh[t], last_sl[t] = b, s, sh, sl
    return pd.DataFrame({"bias": bias, "since": since, "last_sh": last_sh, "last_sl": last_sl},
                        index=df.index)


def rolling_poc(df, window, bins=40):
    """Time-based point of control: price level touched by the most candles in the last `window`."""
    hi, lo = df["High"].values, df["Low"].values
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        h, l = hi[i - window + 1:i + 1], lo[i - window + 1:i + 1]
        pmin, pmax = l.min(), h.max()
        if pmax <= pmin:
            continue
        edges = np.linspace(pmin, pmax, bins + 1)
        cover = ((h[:, None] >= edges[None, :-1]) & (l[:, None] <= edges[None, 1:])).sum(axis=0)
        idx = np.flatnonzero(cover == cover.max())
        mid = idx[len(idx) // 2]
        out[i] = (edges[mid] + edges[mid + 1]) / 2
    return pd.Series(out, index=df.index)


# ---------- data ----------
def get_1h(ticker, days=250):
    """Free hourly data from Yahoo Finance (max ~730 days). Drops the still-forming candle."""
    import yfinance as yf  # imported here so the rest can be tested offline

    df = yf.download(ticker, period=f"{min(int(days), 729)}d", interval="1h",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    now = pd.Timestamp.now(tz=df.index.tz)
    if df.index[-1] + pd.Timedelta("1h") > now:
        df = df.iloc[:-1]
    return df


def resample(df, rule):
    return (df.resample(rule)
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna())


def _entry_frame(h1, c):
    return h1.copy() if c["entry"] == "1h" else resample(h1, c["entry"])


# ---------- strategy 1: EMA pullback ----------
def _prepare_ema(h1, mode):
    c = MODES[mode]
    e = _entry_frame(h1, c)
    t = resample(h1, c["trend"])
    t["ef"] = ema(t["Close"], c["fast"])
    t["es"] = ema(t["Close"], c["slow"])
    t["adx"] = adx(t)
    t["trend"] = np.where(t["adx"] >= c["adx_min"], np.where(t["ef"] > t["es"], 1, -1), 0)
    e["trend"] = t["trend"].shift(1).reindex(e.index, method="ffill").fillna(0)

    e["ema21"] = ema(e["Close"], 21)
    e["rsi"] = rsi(e["Close"])
    e["atr"] = atr(e)

    low_touch = (e["Low"] <= e["ema21"]).astype(int).rolling(6).max() == 1
    high_touch = (e["High"] >= e["ema21"]).astype(int).rolling(6).max() == 1
    buy = ((e["trend"] == 1) & low_touch & (e["rsi"].shift(1) < 50)
           & (e["rsi"] >= 50) & (e["Close"] > e["ema21"]))
    sell = ((e["trend"] == -1) & high_touch & (e["rsi"].shift(1) > 50)
            & (e["rsi"] <= 50) & (e["Close"] < e["ema21"]))
    e["signal"] = np.where(buy, 1, np.where(sell, -1, 0))
    e["sl_px"] = np.nan  # ATR-based exits
    return e


# ---------- strategy 2: BOS + POC ----------
def _prepare_poc(h1, mode):
    c = MODES[mode]
    e = _entry_frame(h1, c)
    t = resample(h1, c["trend"])

    htf = structure(t, c["pivot"])
    e["htf_bias"] = htf["bias"].shift(1).reindex(e.index, method="ffill").fillna(0)

    st = structure(e, c["pivot"])
    for col in ["bias", "since", "last_sh", "last_sl"]:
        e[col] = st[col].values
    e["atr"] = atr(e)
    e["rsi"] = rsi(e["Close"])
    e["poc"] = rolling_poc(e, c["poc_window"])

    tol = POC_TOL * e["atr"]
    close = e["Close"]
    buy = ((e["bias"] == 1) & (e["htf_bias"] == 1) & (e["since"] <= c["bos_window"])
           & (e["Low"] <= e["poc"] + tol) & (close > e["poc"]) & (close > e["Open"])
           & (e["Low"] > e["last_sl"]))
    sell = ((e["bias"] == -1) & (e["htf_bias"] == -1) & (e["since"] <= c["bos_window"])
            & (e["High"] >= e["poc"] - tol) & (close < e["poc"]) & (close < e["Open"])
            & (e["High"] < e["last_sh"]))

    sl_buy = np.minimum(e["Low"].rolling(5).min() - 0.25 * e["atr"], close - 1.0 * e["atr"])
    sl_sell = np.maximum(e["High"].rolling(5).max() + 0.25 * e["atr"], close + 1.0 * e["atr"])
    buy &= (close - sl_buy) <= 3.0 * e["atr"]
    sell &= (sl_sell - close) <= 3.0 * e["atr"]

    e["signal"] = np.where(buy, 1, np.where(sell, -1, 0))
    e["sl_px"] = np.where(buy, sl_buy, np.where(sell, sl_sell, np.nan))
    return e


def prepare(h1, mode):
    """Entry-timeframe candles with indicators and signal (1 buy, -1 sell, 0 none)."""
    return _prepare_poc(h1, mode) if MODES[mode]["kind"] == "poc" else _prepare_ema(h1, mode)


def closed_only(e, h1, mode):
    """Keep only entry candles that have fully closed. Returns (candles, time of last closed hour)."""
    delta = pd.Timedelta(MODES[mode]["delta"])
    h1_end = h1.index[-1] + pd.Timedelta("1h")
    return e[e.index + delta <= h1_end], h1_end
