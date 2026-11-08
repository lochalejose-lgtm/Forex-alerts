"""
Backtest the trend-pullback strategy on the last N days across all 28 forex pairs.

  python backtest.py --days 7 --mode all
  python backtest.py --days 90 --mode poc --spread 2 --email

Assumptions (be aware of them):
- Signal on a closed candle -> enter at the NEXT candle's open.
- If stop and target are both touched in the same candle, the STOP is assumed hit first (conservative).
- Spread cost is deducted from every trade (default 1.5 pips). Swaps/commissions are ignored.
- One open trade per pair at a time. Unfinished trades are closed at the last price (marked OPEN).
Results are in R units: +2R = a full winner, -1R = a full loser.
"""
import argparse
import os
import smtplib
import ssl
from email.message import EmailMessage

import numpy as np
import pandas as pd

from strategy import (PAIRS, MODES, SL_ATR, TP_ATR, POC_RR, get_1h, prepare,
                      closed_only, resolve_modes)


def simulate(e, pair, mode, start, spread_pips):
    c = MODES[mode]
    pip = 0.01 if "JPY" in pair else 0.0001
    spread = spread_pips * pip
    O, H, L, C, A, S = (e[x].values for x in ["Open", "High", "Low", "Close", "atr", "signal"])
    SLPX = e["sl_px"].values
    times, n = e.index, len(e)
    out, busy = [], -1

    for i in np.where(times >= start)[0]:
        if i <= busy or i + 1 >= n or S[i] == 0 or np.isnan(A[i]):
            continue
        side = int(S[i])
        entry = O[i + 1]
        if not np.isnan(SLPX[i]):          # structure-based stop (BOS+POC modes)
            sl = SLPX[i]
            risk = side * (entry - sl)
            if risk <= 0 or risk > 4 * A[i]:
                continue
            tp = entry + side * POC_RR * risk
        else:                              # ATR-based stop (EMA modes)
            risk = SL_ATR * A[i]
            sl = entry - side * risk
            tp = entry + side * TP_ATR * A[i]
        end = min(n, i + 1 + c["max_hold"])

        res, exit_p = None, None
        for j in range(i + 1, end):
            hit_sl = L[j] <= sl if side == 1 else H[j] >= sl
            hit_tp = H[j] >= tp if side == 1 else L[j] <= tp
            if hit_sl:
                res, exit_p = "SL", sl
                break
            if hit_tp:
                res, exit_p = "TP", tp
                break
        if res is None:
            j = end - 1
            exit_p = C[j]
            res = "TIMEOUT" if i + 1 + c["max_hold"] <= n else "OPEN"

        r = side * (exit_p - entry) / risk - spread / risk
        out.append({"time": times[i], "pair": pair, "mode": c["label"],
                    "side": "BUY" if side == 1 else "SELL", "entry": entry,
                    "sl": sl, "tp": tp, "result": res, "R": round(float(r), 2)})
        busy = j
    return out


def stats(df):
    if df.empty:
        return {"trades": 0, "win%": 0.0, "totalR": 0.0, "avgR": 0.0, "PF": 0.0, "maxDD_R": 0.0}
    wins = df.loc[df["R"] > 0, "R"].sum()
    losses = -df.loc[df["R"] <= 0, "R"].sum()
    eq = df.sort_values("time")["R"].cumsum()
    dd = (np.maximum(eq.cummax(), 0) - eq).max()
    return {"trades": len(df), "win%": round(100 * (df["R"] > 0).mean(), 1),
            "totalR": round(df["R"].sum(), 2), "avgR": round(df["R"].mean(), 2),
            "PF": round(wins / losses, 2) if losses > 0 else float("inf"),
            "maxDD_R": round(float(dd), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--mode", default="all",
                    help="1h, 4h, poc1h, poc4h, ema, poc or all")
    ap.add_argument("--spread", type=float, default=1.5, help="spread in pips")
    ap.add_argument("--email", action="store_true")
    a = ap.parse_args()

    modes = resolve_modes(a.mode)
    data = {}
    for pair, tk in PAIRS.items():
        try:
            h1 = get_1h(tk, a.days + 230)
            if h1 is not None and len(h1) > 900:
                data[pair] = h1
            else:
                print(f"{pair}: not enough data, skipped")
        except Exception as ex:
            print(f"{pair}: download error {ex}")

    lines = [f"BACKTEST: last {a.days} days | {len(data)} pairs | spread {a.spread} pips | "
             f"SL {SL_ATR}xATR TP {TP_ATR}xATR\n"]
    all_trades = []
    for mode in modes:
        trades = []
        for pair, h1 in data.items():
            e = prepare(h1, mode)
            e, _ = closed_only(e, h1, mode)
            start = e.index[-1] - pd.Timedelta(days=a.days)
            trades += simulate(e, pair, mode, start, a.spread)
        df = pd.DataFrame(trades)
        all_trades.append(df)
        label = MODES[mode]["label"]
        lines.append(f"===== {label} MODE =====")
        if df.empty:
            lines.append("No trades triggered in this period.\n")
            continue
        s = stats(df)
        lines.append(f"Trades {s['trades']} | Win rate {s['win%']}% | Total {s['totalR']}R | "
                     f"Avg {s['avgR']}R | Profit factor {s['PF']} | Max drawdown {s['maxDD_R']}R")
        per = (df.groupby("pair")["R"].agg(trades="count", win_pct=lambda x: round(100 * (x > 0).mean()),
                                           totalR="sum").round(2).sort_values("totalR", ascending=False))
        lines.append("\nPer pair:\n" + per.to_string() + "\n")

    text = "\n".join(lines)
    print(text)
    with open("results.txt", "w") as f:
        f.write(text)
    if any(not d.empty for d in all_trades):
        pd.concat(all_trades).to_csv("trades.csv", index=False)

    if a.email:
        msg = EmailMessage()
        msg["From"], msg["To"] = os.environ["SMTP_USER"], os.environ["MAIL_TO"]
        msg["Subject"] = f"Forex backtest results ({a.days} days)"
        msg.set_content(text)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            s.send_message(msg)
        print("Results emailed")


if __name__ == "__main__":
    main()
