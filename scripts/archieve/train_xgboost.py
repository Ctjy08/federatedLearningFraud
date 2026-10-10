
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

# 1. Load PaySim
path = Path("data/raw/paysim.csv")

df = pd.read_csv(
    path,
    usecols=["step", "type", "amount", "isFraud"],
    dtype={
        "step": "int16",
        "type": "category",
        "amount": "float32",
        "isFraud": "int8",
    },
)

print("Dataset loaded:", len(df), "transactions")

# 2. Prepare numerical features
X = pd.get_dummies(
    df[["step", "type", "amount"]],
    columns=["type"],
    dtype="float32",
)

X["amount"] = np.log1p(X["amount"])
X = X.astype("float32")
y = df["isFraud"].astype("int8")

# 3. Chronological split
train_mask = df["step"] <= 520
val_mask = (df["step"] > 520) & (df["step"] <= 632)
test_mask = df["step"] > 632

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test),
]:
    print(
        f"{name}: {len(labels):,} rows, "
        f"{int(labels.sum()):,} fraud"
    )
    assert labels.nunique() == 2, (
        f"{name} must contain both classes"
    )

# 4. Handle class imbalance
negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())
weight = negative / positive

print("Fraud class weight:", round(weight, 2))

# 5. Define XGBoost
model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=weight,
    early_stopping_rounds=20,
    n_jobs=4,
    random_state=42,
)

# 6. Train using training data
print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Best boosting iteration:", model.best_iteration)

# 7. Choose threshold using validation data
val_scores = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_scores
)

f1 = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_threshold = float(thresholds[np.argmax(f1)])
print("Best validation threshold:", best_threshold)

# 8. Evaluate on unseen test data
test_scores = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_scores >= best_threshold
).astype(int)

print("\n=== XGBOOST TEST RESULTS ===")
print(classification_report(
    y_test,
    test_predictions,
    digits=4,
    zero_division=0,
))

ap = average_precision_score(y_test, test_scores)
print("Average Precision (AP):", round(ap, 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))

# 9. Show feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)

# 10. Save model
Path("results").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "features": X.columns.tolist(),
    },
    "results/xgboost_model.joblib",
)

print("\nModel saved successfully!")
