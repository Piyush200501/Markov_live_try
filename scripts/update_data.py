"""
Fetches the latest Nifty 50 price and FII/DII flow data, appends any new
trading days to data/history.json, recomputes the Markov chain statistics
using the exact methodology in src/, and rewrites the embedded REAL_DATA
and LIVE_SIGNAL objects inside docs/index.html.

Run manually:
    python scripts/update_data.py

Runs automatically via .github/workflows/update-live-data.yml on a daily
schedule.

IMPORTANT -- read this before relying on it:
The Nifty 50 price fetch (via yfinance) is reliable; Yahoo Finance's API
is well-maintained and widely used. The FII/DII fetch is NOT as reliable
-- it uses an unofficial NSE endpoint (nseindia.com has no official public
API for this). NSE occasionally changes this endpoint or adds stronger
bot-detection without notice. If this script starts failing on the FII/DII
step, check https://www.nseindia.com/reports/fii-dii manually, compare the
response shape against what fetch_fii_dii_today() expects below, and
update the parsing logic accordingly. This is the one part of the pipeline
that will need occasional maintenance.
"""
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.state_discretization import discretize_returns, estimate_tpm_from_states
from src.institutional_regimes import classify_institutional_regimes
from src.conditional_matrices import build_conditional_tpms

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "history.json")
HTML_PATH = os.path.join(ROOT, "docs", "index.html")
THRESHOLD = 0.003


def load_history():
    with open(DATA_PATH) as f:
        return json.load(f)


def save_history(rows):
    with open(DATA_PATH, "w") as f:
        json.dump(rows, f, indent=1)


def fetch_new_price_rows(last_date_str):
    """Fetch Nifty 50 daily closes since the day after last_date_str."""
    import yfinance as yf

    start = (datetime.strptime(last_date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download("^NSEI", start=start, progress=False)
    if df.empty:
        return []
    rows = []
    for date, row in df.iterrows():
        close = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
        rows.append({"date": date.strftime("%Y-%m-%d"), "close": round(close, 2)})
    return rows


def fetch_fii_dii_today():
    """Fetch recent FII/DII net flow from NSE's unofficial API.
    Returns {date_str: {'fii': float, 'dii': float}}, or {} on any failure
    (the pipeline degrades gracefully -- see merge_new_data)."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/reports/fii-dii",
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get("https://www.nseindia.com/api/fiidiiTradeReact", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"WARNING: FII/DII fetch failed ({e}). Skipping flow update this run.")
        return {}

    result = {}
    for entry in data:
        try:
            date_str = datetime.strptime(entry["date"], "%d-%b-%Y").strftime("%Y-%m-%d")
        except Exception:
            continue
        result.setdefault(date_str, {})
        net = float(entry["buyValue"]) - float(entry["sellValue"])
        category = entry.get("category", "")
        if "FII" in category or "FPI" in category:
            result[date_str]["fii"] = round(net, 2)
        elif "DII" in category:
            result[date_str]["dii"] = round(net, 2)
    return result


def merge_new_data(history):
    last_date = history[-1]["date"]
    new_price_rows = fetch_new_price_rows(last_date)
    flow_by_date = fetch_fii_dii_today()

    if not new_price_rows:
        print("No new trading days found.")
        return history, False

    prev_close = history[-1]["close"]
    added = 0
    for row in new_price_rows:
        d = row["date"]
        flow = flow_by_date.get(d, {})
        if "fii" not in flow or "dii" not in flow:
            print(f"No FII/DII figure available yet for {d} -- skipping until it's published.")
            continue
        log_return = round(math.log(row["close"] / prev_close), 6)
        history.append(
            {
                "date": d,
                "close": row["close"],
                "log_return": log_return,
                "fii": flow["fii"],
                "dii": flow["dii"],
                "net_flow": round(flow["fii"] + flow["dii"], 2),
                "regime": None,
            }
        )
        prev_close = row["close"]
        added += 1

    if added == 0:
        return history, False

    flows = [r["net_flow"] for r in history]
    regimes = classify_institutional_regimes(flows)
    for r, regime in zip(history, regimes):
        r["regime"] = regime

    print(f"Added {added} new trading day(s): {history[-added]['date']} to {history[-1]['date']}")
    return history, True


def recompute_and_write(history):
    returns = [r["log_return"] for r in history]
    regimes = [r["regime"] for r in history]

    sweep = {}
    for t in [0.0020, 0.0025, 0.0030, 0.0035, 0.0040]:
        states = discretize_returns(returns, t)
        tpm = estimate_tpm_from_states(states)
        counts = {1: 0, 2: 0, 3: 0}
        for s in states:
            counts[s] += 1
        total = len(states)
        sweep[f"{t:.4f}"] = {
            "tpm": [[round(p, 4) for p in row] for row in tpm],
            "dist": [round(counts[1] / total, 4), round(counts[2] / total, 4), round(counts[3] / total, 4)],
        }

    states_030 = discretize_returns(returns, THRESHOLD)
    conditional = build_conditional_tpms(states_030, regimes)
    cond_out = {k: [[round(p, 4) for p in row] for row in v] for k, v in conditional.items()}

    sums = {1: 0.0, 2: 0.0, 3: 0.0}
    counts = {1: 0, 2: 0, 3: 0}
    for s, r in zip(states_030, returns):
        sums[s] += r
        counts[s] += 1
    expected_returns = [round(sums[s] / counts[s], 6) if counts[s] else 0.0 for s in [1, 2, 3]]

    import numpy as np

    flows = [r["net_flow"] for r in history]
    p25 = round(float(np.percentile(flows, 25)), 2)
    p75 = round(float(np.percentile(flows, 75)), 2)

    latest = history[-1]
    latest_state = states_030[-1]
    latest_regime = regimes[-1]
    forecast_row = cond_out[latest_regime][latest_state - 1]
    live_signal = {
        "date": latest["date"],
        "state": latest_state,
        "regime": latest_regime,
        "forecast": forecast_row,
    }

    real_data = {
        "sweep": sweep,
        "conditional": cond_out,
        "expectedReturns": expected_returns,
        "p25Flow": p25,
        "p75Flow": p75,
        "nDays": len(history),
        "dateStart": history[0]["date"],
        "dateEnd": history[-1]["date"],
    }

    with open(HTML_PATH) as f:
        html = f.read()

    new_real_data_js = "const REAL_DATA = " + json.dumps(real_data, indent=2) + ";"
    html, n1 = re.subn(r"const REAL_DATA = \{.*?\n \};", new_real_data_js, html, count=1, flags=re.S)
    if n1 == 0:
        raise RuntimeError("Could not find REAL_DATA block in docs/index.html -- check the file wasn't hand-edited into a shape this regex can't match.")

    new_live_signal_js = "const LIVE_SIGNAL = " + json.dumps(live_signal, indent=2) + ";"
    html, n2 = re.subn(r"const LIVE_SIGNAL = \{.*?\n \};", new_live_signal_js, html, count=1, flags=re.S)
    if n2 == 0:
        raise RuntimeError("Could not find LIVE_SIGNAL block in docs/index.html.")

    with open(HTML_PATH, "w") as f:
        f.write(html)

    print(f"Updated docs/index.html: {len(history)} days, {history[0]['date']} to {history[-1]['date']}")
    print(f"Live signal: {live_signal}")
    return live_signal


def main():
    history = load_history()
    history, changed = merge_new_data(history)
    if changed:
        save_history(history)
    recompute_and_write(history)


if __name__ == "__main__":
    main()
