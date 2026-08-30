from typing import Dict, List, Sequence
import numpy as np


def calculate_top2_accuracy(
    test_states: Sequence[int],
    test_regimes: Sequence[str],
    conditional_matrices: Dict[str, List[List[float]]],
) -> float:
    hits = 0
    n = len(test_states) - 1
    for t in range(n):
        current_state = test_states[t]
        current_regime = test_regimes[t]
        actual_next_state = test_states[t + 1]
        P_cond = conditional_matrices[current_regime]
        predicted_probs = P_cond[current_state - 1]
        top_2_indices = np.argsort(predicted_probs)[-2:]
        top_2_states = [idx + 1 for idx in top_2_indices]
        if actual_next_state in top_2_states:
            hits += 1
    return (hits / n) * 100 if n > 0 else 0.0
