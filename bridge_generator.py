import numpy as np
import pandas as pd
from data_loader import DUMMY_AADT, DUMMY_N_FT

def generate_synthetic_bridges(n=200, seed=42):
    np.random.seed(seed)
    regions = list(DUMMY_AADT.keys())
    bridges = []
    for i in range(n):
        region = np.random.choice(regions)
        weather_cycle = max(1, int(np.random.normal(DUMMY_N_FT[region], 5)))
        aadt = max(1000, int(np.random.normal(DUMMY_AADT[region], DUMMY_AADT[region] * 0.15)))
        eta = np.random.uniform(25.0, 45.0)
        beta = np.random.uniform(2.0, 3.5)
        bridges.append({
            "bridge_id": i,
            "region": region,
            "AADT": aadt,
            "freeze_thaw_cycles": weather_cycle,
            "eta": eta,
            "beta": beta
        })
    return pd.DataFrame(bridges)

if __name__ == "__main__":
    df = generate_synthetic_bridges(200)
    print(df.head())