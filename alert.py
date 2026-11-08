"""
Live scanner: checks all 28 pairs on four strategies (1H/4H EMA and 1H/4H BOS+POC)
and sends ONE email listing new signals. Run hourly in the cloud (GitHub Actions).
A signal is only reported right after its candle closes, so you don't get duplicates.

  python alert.py              # all four strategies
  python alert.py --mode poc   # only BOS+POC (1H and 4H)
  python alert.py --mode 4h    # a single one: 1h, 4h, poc1h, poc4h
  python alert.py --test       # send a test email

Env vars: SMTP_USER, SMTP_PASS (Gmail app password), MAIL_TO
"""
import os
import sys
import smtplib
import ssl
from email.message import EmailMessage

import pandas as pd

from strategy import (PAIRS, MODES, SL_ATR, TP_ATR, POC_RR, get_1h, prepare,
                      closed_only, resolve_modes)


def send_email(subject, body):
    user, pwd, to = os.environ["SMTP_USER"], os.environ["SMTP_PASS"], os.environ["MAIL_TO"]
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
        s.login(user, pwd)
        s.send_message(msg)


def describe(pair, mode, last, h1_end):
    side = 1 if last["signal"] == 1 else -1
    entry, a = float(last["Close"]), float(last["atr"])
    dec = 3 if "JPY" in pair else 5
    if pd.notna(last["sl_px"]):
        sl = float(last["sl_px"])
        tp = entry + side * POC_RR * abs(entry - sl)
    else:
        sl = entry - side * SL_ATR * a
        tp = entry + side * TP_ATR * a
    rr = abs(tp - entry) / abs(entry - sl)
    if MODES[mode]["kind"] == "poc":
        why = (f"{'Bullish' if side == 1 else 'Bearish'} BOS + HTF agrees, "
               f"price rejected POC {float(last['poc']):.{dec}f}")
    else:
        why = f"Trend pullback, RSI {float(last['rsi']):.1f}"
    return (f"{'BUY' if side == 1 else 'SELL'} {pair} [{MODES[mode]['label']}]\n"
            f"  Entry ~ {entry:.{dec}f}\n"
            f"  Stop loss   {sl:.{dec}f}\n"
            f"  Take profit {tp:.{dec}f}  (R:R 1:{rr:.1f})\n"
            f"  Why: {why}\n"
            f"  Candle closed {h1_end}\n")


def main():
    if "--test" in sys.argv:
        send_email("Forex bot test", "Email alerts are working.")
        print("Test email sent")
        return

    modes = resolve_modes("all")
    if "--mode" in sys.argv:
        modes = resolve_modes(sys.argv[sys.argv.index("--mode") + 1])

    found = []
    for pair, tk in PAIRS.items():
        try:
            h1 = get_1h(tk, 250)
            if h1 is None or len(h1) < 900:
                print(f"{pair}: not enough data")
                continue
            for mode in modes:
                e = prepare(h1, mode)
                e, h1_end = closed_only(e, h1, mode)
                if e.empty:
                    continue
                delta = pd.Timedelta(MODES[mode]["delta"])
                last = e.iloc[-1]
                # only alert for the candle that JUST closed
                if e.index[-1] + delta != h1_end or last["signal"] == 0:
                    continue
                found.append(describe(pair, mode, last, h1_end))
        except Exception as ex:
            print(f"{pair}: error {ex}")

    if found:
        body = "\n".join(found) + "\nRisk max 1% of your account per trade. Not financial advice."
        send_email(f"Forex signals: {len(found)} new", body)
        print(f"{len(found)} signal(s) emailed")
    else:
        print("No new signals this hour")


if __name__ == "__main__":
    main()
