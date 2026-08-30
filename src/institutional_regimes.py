from typing import List, Sequence
import numpy as np

REGIME_SP, REGIME_N, REGIME_SN = "SP", "N", "SN"


def classify_institutional_regimes(net_flows: Sequence[float]) -> List[str]:
    p25 = np.percentile(net_flows, 25)
    p75 = np.percentile(net_flows, 75)
    regimes = []
    for flow in net_flows:
        if flow > p75:
            regimes.append(REGIME_SP)
        elif flow < p25:
            regimes.append(REGIME_SN)
        else:
            regimes.append(REGIME_N)
    return regimes
