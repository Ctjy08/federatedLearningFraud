
import torch

from flwr.server import ServerApp
from flwr.serverapp import Grid
from flwr.common import Context, ArrayRecord, ConfigRecord
from flwr.serverapp.strategy import FedAvg

from crossbank.models.mlp import FraudMLP


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    torch.manual_seed(42)

    # 1. Initialise an untrained global neural network.
    global_model = FraudMLP(input_size=9)
    initial_weights = ArrayRecord(
        global_model.state_dict()
    )

    # 2. Standard Federated Averaging.
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=3,
        weighted_by_key="num-examples",
    )

    rounds = int(context.run_config["num-server-rounds"])

    print(f"Starting federated training: {rounds} rounds")

    # 3. Distribute, train, aggregate, repeat.
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_weights,
        train_config=ConfigRecord({
            "learning-rate": float(
                context.run_config["learning-rate"]
            ),
        }),
        num_rounds=rounds,
    )

    # 4. Save the final global model weights.
    torch.save(
        result.arrays.to_torch_state_dict(),
        "global_model.pt",
    )

    print("\nFederated training finished!")
    print("Global model saved as global_model.pt")
