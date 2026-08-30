import random
from typing import List, Sequence, Tuple


def run_mcmc_simulation(
    start_state: int,
    P_matrix: Sequence[Sequence[float]],
    expected_returns: Sequence[float],
    days: int = 30,
) -> Tuple[List[int], List[float]]:
    simulated_states = [start_state]
    simulated_returns: List[float] = []
    for _ in range(days):
        current_idx = simulated_states[-1] - 1
        probs = P_matrix[current_idx]
        r = random.uniform(0, 1)
        cumulative_prob = 0.0
        next_state = simulated_states[-1]
        for j, p in enumerate(probs):
            cumulative_prob += p
            if r <= cumulative_prob:
                next_state = j + 1
                break
        simulated_states.append(next_state)
        simulated_returns.append(expected_returns[next_state - 1])
    return simulated_states, simulated_returns
