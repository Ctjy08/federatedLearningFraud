
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
)

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)
from crossbank.models.mlp import FraudMLP

torch.manual_seed(42)
np.random.seed(42)

FEATURES = [
    "step", "amount", "oldbalanceOrg", "oldbalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER"
]

def load_data(path):
    df = pd.read_csv(
        path,
        usecols=[
            "step", "type", "amount",
            "oldbalanceOrg", "oldbalanceDest", "isFraud"
        ],
    )

    x = pd.DataFrame(index=df.index)
    x["step"] = df["step"].astype("float32")

    for col in ["amount", "oldbalanceOrg", "oldbalanceDest"]:
        x[col] = np.log1p(
            df[col].clip(lower=0)
        ).astype("float32")

    for kind in ["CASH_IN", "CASH_OUT", "DEBIT",
                 "PAYMENT", "TRANSFER"]:
        x[f"type_{kind}"] = (
            df["type"] == kind
        ).astype("float32")

    y = df["isFraud"].to_numpy(dtype=np.float32)
    return x[FEATURES].to_numpy(dtype=np.float32), y


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bank", choices=["a", "b", "c"], default="a"
)
args = parser.parse_args()

print(f"Loading Bank {args.bank.upper()}...")

X_train, y_train = load_data(
    f"data/processed/bank_{args.bank}.csv"
)
X_val, y_val = load_data("data/processed/validation.csv")

# Fit preprocessing ONLY on the selected bank.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_val = scaler.transform(X_val).astype("float32")

train_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    ),
    batch_size=1024,
    shuffle=True
)

model = FraudMLP(input_size=len(FEATURES))

positive = float(y_train.sum())
negative = float(len(y_train) - positive)

loss_fn = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([negative / positive])
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Training rows: {len(y_train):,}")
print(f"Fraud cases: {int(positive):,}")

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for features, labels in train_loader:
        optimizer.zero_grad()

        logits = model(features)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(y_train):.6f}"
    )

# Evaluate without updating model weights.
model.eval()

with torch.no_grad():
    val_logits = model(
        torch.from_numpy(X_val)
    )
    scores = torch.sigmoid(val_logits).numpy()

precision, recall, thresholds = precision_recall_curve(
    y_val, scores
)

f1 = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)
threshold = float(thresholds[np.argmax(f1)])

predictions = (scores >= threshold).astype(int)

print("\n=== VALIDATION RESULTS ===")
print("Selected threshold:", round(threshold, 4))
print(classification_report(
    y_val, predictions, digits=4, zero_division=0
))
print(
    "Average Precision:",
    round(average_precision_score(y_val, scores), 4)
)

Path("results").mkdir(exist_ok=True)
torch.save(
    {
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": FEATURES,
        "threshold": threshold,
    },
    f"results/bank_{args.bank}_pytorch.pt"
)

print("Local model saved successfully!")
