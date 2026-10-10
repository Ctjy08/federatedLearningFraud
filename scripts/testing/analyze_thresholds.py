
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

saved = joblib.load("results/xgboost_model.joblib")
model = saved["model"]

df = pd.read_csv(
    "data/raw/paysim.csv",
    usecols=["step", "type", "amount", "isFraud"],
)

# Use validation data ONLY for threshold selection.
val = df[(df["step"] > 520) & (df["step"] <= 632)]

X = pd.get_dummies(
    val[["step", "type", "amount"]],
    columns=["type"],
    dtype="float32",
)

X["amount"] = np.log1p(X["amount"])
X = X.reindex(
    columns=saved["features"],
    fill_value=0,
).astype("float32")

y = val["isFraud"].astype(int)
scores = model.predict_proba(X)[:, 1]

print("\nTHRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1 | False positives")

for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.9175]:
    predictions = (scores >= threshold).astype(int)

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:8.4f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:5.4f} | {fp}"
    )

print("\nPredicted score distribution:")
print(pd.Series(scores).describe(
    percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
))

print("\nScores for actual fraud:")
print(pd.Series(scores[y.to_numpy() == 1]).describe())

print("\nScores for legitimate transactions:")
print(pd.Series(scores[y.to_numpy() == 0]).describe())