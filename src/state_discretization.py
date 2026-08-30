from typing import List

STATE_UP, STATE_DOWN, STATE_STABLE = 1, 2, 3
N_STATES = 3


def discretize_returns(returns: List[float], threshold: float = 0.003) -> List[int]:
    states = []
    for r in returns:
        if r > threshold:
            states.append(STATE_UP)
        elif r < -threshold:
            states.append(STATE_DOWN)
        else:
            states.append(STATE_STABLE)
    return states


def estimate_tpm_from_states(states: List[int], n_states: int = N_STATES) -> List[List[float]]:
    counts = [[0] * n_states for _ in range(n_states)]
    for t in range(len(states) - 1):
        i, j = states[t] - 1, states[t + 1] - 1
        counts[i][j] += 1
    P = [[0.0] * n_states for _ in range(n_states)]
    for i in range(n_states):
        row_sum = sum(counts[i])
        if row_sum > 0:
            for j in range(n_states):
                P[i][j] = counts[i][j] / row_sum
    return P


def calculate_baseline_tpm(returns: List[float], threshold: float = 0.003) -> List[List[float]]:
    states = discretize_returns(returns, threshold)
    return estimate_tpm_from_states(states)
