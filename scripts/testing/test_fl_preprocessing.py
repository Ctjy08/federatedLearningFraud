from pathlib import Path
import sys
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from crossbank.data.fl_preprocessing import load_fl_data

for bank in ["a", "b", "c"]:
    X, y = load_fl_data(
        f"data/processed/bank_{bank}.csv"
    )

    assert X.shape[1] == 9
    assert np.isfinite(X).all()
    assert set(np.unique(y)).issubset({0.0, 1.0})

    print(f"\nBank {bank.upper()}")
    print("Transactions:", len(X))
    print("Features:", X.shape[1])
    print("Fraud cases:", int(y.sum()))
    print("First transaction features:", X[0])

print("\nAll banks use consistent preprocessing!")