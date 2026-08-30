"""
End-to-end pipeline on REAL Nifty 50 + FII/DII data (CSV export format).
Usage: python examples/demo_real_data.py path/to/your_data.csv
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Needs pandas and numpy: pip install pandas numpy --break-system-packages")

from src.state_discretization import discretize_returns, estimate_tpm_from_states
from src.conditional_matrices import build_conditional_tpms
from src.validation import calculate_top2_accuracy
from src.monte_carlo_simulation import run_mcmc_simulation


def load_data(csv_path: str) -> "pd.DataFrame":
    df = pd.read_csv(csv_path)
    df["Price"] = df["Price"].astype(str).str.replace(",", "").astype(float)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    if df["Date"].isna().any():
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df["Price"] / df["Price"].shift(1))
    return df.dropna(subset=["log_return"]).reset_index(drop=True)


def classify_flow_regimes(net_flows) -> list:
    p25, p75 = np.percentile(net_flows, 25), np.percentile(net_flows, 75)
    return ["SP" if f > p75 else "SN" if f < p25 else "N" for f in net_flows]


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python examples/demo_real_data.py path/to/your_data.csv")

    df = load_data(sys.argv[1])
    threshold = 0.003
    returns = df["log_return"].tolist()
    if "Net Institutional Flow" in df.columns:
        regimes = classify_flow_regimes(df["Net Institutional Flow"].tolist())
    else:
        regimes = classify_flow_regimes(df["FII"] + df["DII"])
    states = discretize_returns(returns, threshold)

    print(f"Loaded {len(df)} trading days: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

    print("Threshold sensitivity (Up / Down / Stagnant persistence):")
    for t in [0.0020, 0.0025, 0.0030, 0.0035, 0.0040]:
        s = discretize_returns(returns, t)
        tpm = estimate_tpm_from_states(s)
        print(f"  {t*100:.2f}%:  p11={tpm[0][0]*100:5.2f}%  p22={tpm[1][1]*100:5.2f}%  p33={tpm[2][2]*100:5.2f}%")

    baseline_tpm = estimate_tpm_from_states(states)
    print(f"\nBaseline TPM at {threshold*100:.2f}%:")
    for row in baseline_tpm:
        print("  ", ["%.4f" % p for p in row])

    conditional = build_conditional_tpms(states, regimes)
    print("\nConditional TPMs by institutional regime:")
    for label in ["SN", "N", "SP"]:
        print(f"  {label}:")
        for row in conditional[label]:
            print("    ", ["%.4f" % p for p in row])
        print(f"    -> Down persistence (p22): {conditional[label][1][1]*100:.2f}%")

    split = int(len(states) * 0.8)
    train_cond = build_conditional_tpms(states[:split], regimes[:split])
    acc = calculate_top2_accuracy(states[split:], regimes[split:], train_cond)
    print(f"\nTop-2 Probabilistic Accuracy (80/20 split): {acc:.2f}%")

    sim_path, _ = run_mcmc_simulation(
        start_state=1, P_matrix=baseline_tpm,
        expected_returns=[0.005, -0.005, 0.0], days=30,
    )
    print(f"\n30-day simulated path (from Upward): {sim_path}")


if __name__ == "__main__":
    main()
