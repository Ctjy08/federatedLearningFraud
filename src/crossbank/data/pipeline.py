from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["step", "amount", "oldbalanceOrg", "oldbalanceDest", "type_CASH_OUT", "type_TRANSFER"]
REQUIRED = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"]

def load_features(path: str | Path):
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    x = pd.DataFrame({
        "step": df["step"].astype(float),
        "amount": df["amount"].astype(float),
        "oldbalanceOrg": df["oldbalanceOrg"].astype(float),
        "oldbalanceDest": df["oldbalanceDest"].astype(float),
        "type_CASH_OUT": (df["type"] == "CASH_OUT").astype(float),
        "type_TRANSFER": (df["type"] == "TRANSFER").astype(float),
    })
    y = df["isFraud"].astype(int)
    return df, x, y

def chrono_split(df, test_fraction=.2):
    # Keep equal timestamps on same side where possible.
    cut = int(len(df) * (1 - test_fraction))
    if cut <= 0 or cut >= len(df):
        raise ValueError("Dataset too small")
    split_step = df.iloc[cut]["step"]
    train = df[df["step"] < split_step]
    test = df[df["step"] >= split_step]
    if train.empty or test.empty:
        raise ValueError("Cannot split chronologically; require multiple steps")
    return train, test

def partition_non_iid(train, n_banks=3, seed=42):
    # Partition data into disjoint banks with different amount profiles; avoid label-driven splits.
    rng = np.random.default_rng(seed)
    shuffled = train.copy().sample(frac=1, random_state=seed)
    ranks = shuffled["amount"].rank(pct=True)
    # Mostly low, medium, high transaction amounts by bank, with overlap so all can train.
    prob = np.stack([1.2-ranks, 1 - np.abs(ranks - .5), .2+ranks], axis=1)
    if n_banks != 3:
        raise ValueError("Initial non-IID partition supports exactly 3 banks")
    prob /= prob.sum(axis=1, keepdims=True)
    u = rng.random(len(shuffled))
    assignment = (u[:, None] > prob.cumsum(axis=1)).sum(axis=1)
    return [shuffled.iloc[np.flatnonzero(assignment == i)].copy() for i in range(n_banks)]
