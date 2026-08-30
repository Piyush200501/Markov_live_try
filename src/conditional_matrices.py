from typing import Dict, List, Sequence

N_STATES = 3


def build_conditional_tpms(states: Sequence[int], regimes: Sequence[str], n_states: int = N_STATES) -> Dict[str, List[List[float]]]:
    if len(states) != len(regimes):
        raise ValueError("states and regimes must be the same length")
    labels = sorted(set(regimes))
    counts = {label: [[0] * n_states for _ in range(n_states)] for label in labels}
    for t in range(len(states) - 1):
        label = regimes[t]
        i, j = states[t] - 1, states[t + 1] - 1
        counts[label][i][j] += 1
    matrices: Dict[str, List[List[float]]] = {}
    for label in labels:
        P = [[0.0] * n_states for _ in range(n_states)]
        for i in range(n_states):
            row_sum = sum(counts[label][i])
            if row_sum > 0:
                for j in range(n_states):
                    P[i][j] = counts[label][i][j] / row_sum
        matrices[label] = P
    return matrices
