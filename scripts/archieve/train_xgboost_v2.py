
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
    roc_auc_score,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = Path("data/raw/paysim.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "xgboost_v2_model.joblib"

COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "isFraud",
]

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading PaySim dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=COLUMNS,
    dtype={
        "step": "int16",
        "type": "category",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "oldbalanceDest": "float32",
        "isFraud": "int8",
    },
)

print(f"Dataset loaded: {len(df):,} transactions")
print(f"Total fraud: {df['isFraud'].sum():,}")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

print("\nPreparing features...")

X = pd.get_dummies(
    df[
        [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
        ]
    ],
    columns=["type"],
    dtype="float32",
)

# Log-transform skewed financial values
X["amount"] = np.log1p(
    df["amount"].clip(lower=0)
)

X["oldbalanceOrg"] = np.log1p(
    df["oldbalanceOrg"].clip(lower=0)
)

X["oldbalanceDest"] = np.log1p(
    df["oldbalanceDest"].clip(lower=0)
)

# Derived features
X["amount_to_balance_ratio"] = (
    df["amount"] /
    (df["oldbalanceOrg"] + 1)
).clip(lower=0, upper=1000)

X["sender_balance_remaining_ratio"] = (
    (df["oldbalanceOrg"] - df["amount"])
    / (df["oldbalanceOrg"] + 1)
).clip(lower=-1000, upper=1000)

X["recipient_balance_ratio"] = (
    df["amount"] /
    (df["oldbalanceDest"] + 1)
).clip(lower=0, upper=1000)

X = X.astype("float32")

# Ground-truth target
y = df["isFraud"].astype("int8")

print("Number of features:", X.shape[1])
print("Features:", X.columns.tolist())

# ==========================================
# 4. CHRONOLOGICAL TRAIN/VAL/TEST SPLIT
# ==========================================

train_mask = df["step"] <= 520

val_mask = (
    (df["step"] > 520) &
    (df["step"] <= 632)
)

test_mask = df["step"] > 632

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[val_mask]
y_val = y.loc[val_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test),
]:
    print(
        f"{name}: {len(labels):,} transactions, "
        f"{int(labels.sum()):,} fraud cases"
    )

    if labels.nunique() != 2:
        raise ValueError(
            f"{name} dataset must contain both classes."
        )

# ==========================================
# 5. HANDLE CLASS IMBALANCE
# ==========================================

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

class_weight = negative / positive

print(f"\nFraud class weight: {class_weight:.2f}")

# ==========================================
# 6. CREATE XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=class_weight,
    early_stopping_rounds=20,
    n_jobs=4,
    random_state=42,
)

# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining XGBoost V2...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Best boosting iteration:", model.best_iteration)

# ==========================================
# 8. OPTIMISE THRESHOLD ON VALIDATION SET
# ==========================================

val_scores = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_scores,
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_index = np.argmax(f1_scores)
best_threshold = float(thresholds[best_index])

print(f"\nBest validation threshold: {best_threshold:.6f}")
print(f"Best validation F1: {f1_scores[best_index]:.4f}")

# ==========================================
# 9. EVALUATE ON TEST SET
# ==========================================

test_scores = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_scores >= best_threshold
).astype(int)

print("\n=== XGBOOST V2 TEST RESULTS ===")

print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0,
    )
)

ap = average_precision_score(
    y_test,
    test_scores,
)

roc_auc = roc_auc_score(
    y_test,
    test_scores,
)

print(f"Average Precision (AP): {ap:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"\nTrue Negatives: {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives: {tp:,}")

# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance)

# ==========================================
# 11. SAVE MODEL AND METRICS
# ==========================================

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "features": X.columns.tolist(),
        "feature_version": "balance_aware_v2",
    },
    MODEL_PATH,
)

metrics = {
    "model": "XGBoost V2",
    "threshold": best_threshold,
    "average_precision": ap,
    "roc_auc": roc_auc,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}

pd.DataFrame([metrics]).to_csv(
    RESULTS_DIR / "xgboost_v2_metrics.csv",
    index=False,
)

importance.to_csv(
    RESULTS_DIR / "xgboost_v2_feature_importance.csv",
    header=["importance"],
)

print("\nModel and metrics saved successfully!")
print("Model:", MODEL_PATH)
