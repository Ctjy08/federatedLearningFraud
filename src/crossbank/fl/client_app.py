
from pathlib import Path
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

from flwr.client import ClientApp
from flwr.common import (
    ArrayRecord,
    MetricRecord,
    RecordDict,
    Message,
    Context,
)

from crossbank.data.fl_preprocessing import load_fl_data
from crossbank.models.mlp import FraudMLP


app = ClientApp()

BANKS = ["a", "b", "c"]

# Adjust if your project folder changes.
DATA_DIR = Path("data/processed")


@app.train()
def train(msg: Message, context: Context) -> Message:

    # 1. Identify which bank this virtual node represents.
    partition_id = int(context.node_config["partition-id"])
    bank = BANKS[partition_id]

    print(f"\nBank {bank.upper()} starting local training...")

    # 2. Create the same neural network architecture.
    model = FraudMLP(input_size=9)

    # 3. Load the latest GLOBAL model weights.
    global_weights = msg.content[
        "arrays"
    ].to_torch_state_dict()

    model.load_state_dict(global_weights)

    # 4. Load ONLY this bank's private transactions.
    X, y = load_fl_data(
        DATA_DIR / f"bank_{bank}.csv"
    )

    # Small reproducible sample for our first test.
    max_samples = int(context.run_config["max-samples"])

    if len(y) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(y),
            size=max_samples,
            replace=False,
        )
        X = X[indices]
        y = y[indices]

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(context.run_config["batch-size"]),
        shuffle=True,
    )

    # 5. Set up local training.
    positive = max(float(y.sum()), 1.0)
    negative = float(len(y) - y.sum())

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negative / positive]
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(context.run_config["learning-rate"]),
    )

    local_epochs = int(context.run_config["local-epochs"])

    # 6. Train the local copy of the global model.
    model.train()
    total_loss = 0.0

    for epoch in range(local_epochs):
        total_loss = 0.0

        for features, labels in loader:
            optimizer.zero_grad()

            logits = model(features)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        avg_loss = total_loss / len(dataset)

        print(
            f"Bank {bank.upper()} | "
            f"Epoch {epoch + 1}/{local_epochs} | "
            f"Loss: {avg_loss:.6f}"
        )

    # 7. Send trained model weights to the server.
    # Raw transaction records are not included.
    arrays = ArrayRecord(model.state_dict())

    metrics = MetricRecord({
        "train_loss": float(avg_loss),
        "num-examples": len(dataset),
    })

    return Message(
        content=RecordDict({
            "arrays": arrays,
            "metrics": metrics,
        }),
        reply_to=msg,
    )
