"""
End-to-end demo of the full Markov chain pipeline on SYNTHETIC data.
Run: python examples/demo.py
"""
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.state_discretization import discretize_returns, calculate_baseline_tpm
from src.institutional_regimes import classify_institutional_regimes
from src.conditional_matrices import build_conditional_tpms
from src.validation import calculate_top2_accuracy
from src.monte_carlo_simulation import run_mcmc_simulation


def generate_synthetic_data(n_days: int = 1600, seed: int = 42):
    random.seed(seed)
    returns = [random.gauss(0.0003, 0.009) for _ in range(n_days)]
    flows = [random.gauss(0, 500) for _ in range(n_days)]
    return returns, flows


def main():
    returns, flows = generate_synthetic_data()
    split = int(len(returns) * 0.85)
    train_returns, test_returns = returns[:split], returns[split:]
    train_flows, test_flows = flows[:split], flows[split:]
    threshold = 0.003

    baseline_tpm = calculate_baseline_tpm(train_returns, threshold=threshold)
    print("Baseline Transition Probability Matrix (Up / Down / Stagnant):")
    for row in baseline_tpm:
        print("  ", ["%.4f" % p for p in row])

    train_states = discretize_returns(train_returns, threshold=threshold)
    train_regimes = classify_institutional_regimes(train_flows)
    conditional_tpms = build_conditional_tpms(train_states, train_regimes)

    print("\nConditional TPMs by institutional flow regime:")
    for label, matrix in conditional_tpms.items():
        print(f"  {label}:")
        for row in matrix:
            print("    ", ["%.4f" % p for p in row])

    test_states = discretize_returns(test_returns, threshold=threshold)
    test_regimes = classify_institutional_regimes(test_flows)
    accuracy = calculate_top2_accuracy(test_states, test_regimes, conditional_tpms)
    print(f"\nTop-2 Probabilistic Accuracy on held-out data: {accuracy:.2f}%")

    expected_returns_per_state = [0.005, -0.005, 0.0]
    sim_states, sim_returns = run_mcmc_simulation(
        start_state=1, P_matrix=baseline_tpm,
        expected_returns=expected_returns_per_state, days=30,
    )
    print(f"\n30-day simulated state path (from Upward): {sim_states}")


if __name__ == "__main__":
    main()
