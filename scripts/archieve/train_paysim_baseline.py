
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

path = Path("data/raw/paysim.csv")

# Load relevant columns
columns = ["step", "type", "amount", "isFraud"]

df = pd.read_csv(path, usecols=columns)

# Create features
X = pd.get_dummies(
    df[["step", "type", "amount"]],
    columns=["type"],
    dtype=int,
)

# Log-transform the heavily skewed transaction amount
X["amount"] = np.log1p(X["amount"])

y = df["isFraud"].astype(int)

# Chronological split (earlier transactions for training)
train_mask = df["step"] <= 520
val_mask = (df["step"] > 520) & (df["step"] <= 632)
test_mask = df["step"] > 632

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

assert y_train.nunique() == 2, "Training needs both classes."
assert y_val.nunique() == 2, "Validation needs both classes."
assert y_test.nunique() == 2, "Test needs both classes."

print("Training transactions:", len(X_train))
print("Validation transactions:", len(X_val))
print("Test transactions:", len(X_test))
print("Fraud in training:", int(y_train.sum()))

# Balance the importance of minority fraud examples
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
    ),
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Choose a decision threshold using validation only
val_scores = model.predict_proba(X_val)[:, 1]

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    y_val, val_scores
)

f1 = (
    2 * precision[:-1] * recall[:-1] /
    (precision[:-1] + recall[:-1] + 1e-12)
)
best_threshold = thresholds[np.argmax(f1)]

# Test once using the selected threshold
test_scores = model.predict_proba(X_test)[:, 1]
test_predictions = (test_scores >= best_threshold).astype(int)

print("\nBest validation threshold:", round(best_threshold, 4))
print("\nTest results:")
print(classification_report(
    y_test, test_predictions,
    digits=4,
    zero_division=0,
))

print(
    "Test PR-AUC (average precision):",
    round(average_precision_score(y_test, test_scores), 4),
)

# Save model
import joblib

Path("results").mkdir(exist_ok=True)
joblib.dump(
    {"model": model, "threshold": float(best_threshold)},
    "results/logistic_regression.joblib",
)

print("\nModel saved!")
