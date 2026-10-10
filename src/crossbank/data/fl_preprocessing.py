
import numpy as np
import pandas as pd

# All banks must use the same features in the same order
FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


def load_fl_data(path):
    """
    Load PaySim transactions and apply identical
    preprocessing for every federated-learning bank.

    Returns:
        X: Transaction features
        y: Fraud labels (0 or 1)
    """

    # 1. Read transaction data
    df = pd.read_csv(
        path,
        usecols=[
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
            "isFraud",
        ],
    )

    # 2. Create a feature table
    x = pd.DataFrame(index=df.index)

    # 3. Scale transaction time
    x["step"] = (
        df["step"].astype("float32") / 743.0
    )

    # 4. Scale transaction amounts and balances
    for column in [
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]:
        x[column] = (
            np.log1p(df[column].clip(lower=0)) / 20.0
        ).astype("float32")

    # 5. Convert transaction types into numbers
    for transaction_type in TRANSACTION_TYPES:
        x[f"type_{transaction_type}"] = (
            df["type"] == transaction_type
        ).astype("float32")

    # 6. Extract features (X)
    X = x[FEATURES].to_numpy(dtype=np.float32)

    # 7. Extract correct fraud labels (y)
    y = df["isFraud"].to_numpy(dtype=np.float32)

    return X, y
