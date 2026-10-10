# Cross-Bank Federated Fraud Detection

**CSC3104 — Cloud & Distributed Computing**  
**Project:** Resilient Cloud-Native Federated Learning for Privacy-Preserving Cross-Bank Fraud Detection

> **Read this first:** This is a research/student prototype using **PaySim synthetic financial transactions**, NOT real bank data. It does not prove that the system is private or production-secure. Federated learning keeps transaction rows at the clients by design, but model updates can still leak information without additional protections.

## 0. What is this project, in plain English?

Imagine **Bank A, Bank B, and Bank C** want one fraud-detection AI model. They cannot simply pool their customer transaction tables.

1. A coordinator creates a **global PyTorch model**.
2. Each bank receives **a copy of the same model weights**.
3. Each bank trains that copy using **its own transactions**.
4. Banks return **updated model weights**, not the raw transaction tables.
5. Flower combines the weights using **FedAvg** and distributes the updated global model.
6. Repeat for several rounds.

Our later original contribution will be **fault/straggler-tolerant coordination** (for example, quorum-based aggregation, client timeouts, and possibly staleness-aware updates), evaluated against ordinary FedAvg. **Those resilience mechanisms are planned, not already implemented.**

### What are the tools for?

| Tool | Job |
|---|---|
| Python | Programming language |
| PyTorch | Neural network, loss function, local training |
| Adam | Updates neural-network weights during local training |
| Flower | Coordinates federated-learning rounds and FedAvg |
| Docker | Packages programs into container images (later stage) |
| Kubernetes | Runs/manages networked containers, including failures (later stage) |

**Important:** `train_bank.py` and `train_centralized.py` are **comparison experiments**. The actual collaborative system is the **Flower client + server**. Do **not** average the older `bank_a_pytorch.pt`, `bank_b_pytorch.pt`, and `bank_c_pytorch.pt` checkpoints: those independent models used bank-specific scalers, so their weights are not directly compatible.

---

### Step 1.2 — Install Python 3.12

Install Python 3.12 from <https://www.python.org/downloads/>. Python **3.12** is the version tested for the existing development environment; do not accidentally use a separate system Python 3.14 installation.

Open **PowerShell** and verify:

```powershell
py -3.12 --version
```

Expected: `Python 3.12.x`. If the command fails, check the Python installation before continuing.

### Step 1.3 — Install `uv` (Python environment/package manager)

Follow the Windows instructions at <https://docs.astral.sh/uv/getting-started/installation/>. One common Windows PowerShell installation command is:

```powershell
winget install --id=astral-sh.uv -e
```

Close and reopen PowerShell, then check:

```powershell
uv --version
```


## 2. Create and activate the virtual environment

A virtual environment keeps this project's Python packages separate from other projects.

From the **project root**:

```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Expected: `Python 3.12.x`. Your prompt may show `(.venv)`.

If PowerShell blocks activation, a session-only workaround is:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `.venv` already exists and works, **do not recreate it unnecessarily**.

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

When `flDetection.toml` is present and correctly configured (see **Section 8**), also run:

```powershell
uv pip install -e .
```

This installs **your own `crossbank` Python package in editable mode**, so imports such as `from crossbank.models.mlp import FraudMLP` work without manually editing `sys.path`.


---



## 4. Get the PaySim dataset

We are **not using the old `generate_demo_data.py` anymore**.

1. Open <https://www.kaggle.com/datasets/moonknightmarvel/paysim>.
2. Log in to Kaggle if necessary.
3. Download the dataset ZIP.
4. Extract the CSV file. Kaggle's file name may differ from `paysim.csv`.
5. Rename the extracted transaction CSV to **`paysim.csv`**.
6. Place it in **`crossbank-fl/data/raw/paysim.csv`**.


**Dataset note:** PaySim is simulated mobile-money transaction data. It is **not** a dataset of actual banks, actual money-laundering layering, or real customer identities. Our three "banks" are artificial partitions for a federated-learning experiment.

---

## 5. Prepare Bank A, Bank B and Bank C

### What this step does

The preparation code splits PaySim **chronologically** and partitions the training portion into three artificial banks with different transaction-amount distributions and fraud rates (**non-IID data**).

The intended prepared files are:

| File | Purpose |
|---|---|
| `bank_a.csv` | Bank A private training data |
| `bank_b.csv` | Bank B private training data |
| `bank_c.csv` | Bank C private training data |
| `validation.csv` | Shared held-out development/validation split |
| `test.csv` | Final evaluation split |



---

## 6. Verify common FL preprocessing (do this BEFORE Flower)

Every bank must use **exactly the same input features and scaling** for FedAvg to make sense.

**Code location:** `src/crossbank/data/fl_preprocessing.py`  
**Test location:** `scripts/testing/test_fl_preprocessing.py`.

The current intended feature set has **nine columns**:

1. `step`
2. `amount`
3. `oldbalanceOrg`
4. `oldbalanceDest`
5. `type_CASH_IN`
6. `type_CASH_OUT`
7. `type_DEBIT`
8. `type_PAYMENT`
9. `type_TRANSFER`

The preprocessing uses the same fixed transforms across banks (for example, scaled `step`, log-scaled amounts/balances and one-hot transaction types). **It does not fit a separate `StandardScaler` for each bank.**


---

## 7. Know which training programs are which

| Script / files | What they really do | Needed for final system? |
|---|---|---|
| `train_bank.py` | Train **three independent** PyTorch models (A, B, C). No federation. | Useful comparison |
| `train_centralized.py` | Train **one** PyTorch model from pooled A+B+C training transactions. | Important benchmark; planned/updating |
| Old `train_baselines.py` | Trains Logistic Regression + a centralised MLP using the earlier pipeline; can be heavy and may use inconsistent preprocessing. | Keep as archive/reference, not the final fair benchmark |
| `train_xgboost*.py` | Earlier model/feature experiments. | Optional archive |
| `run_federated.py` | **Original starter's single-process FedAvg simulation**; not networked bank services. | Learning/debug reference |
| `src/crossbank/fl/client_app.py` | Flower program that trains the shared model using one bank's local data. | **Yes — implement/test** |
| `src/crossbank/fl/server_app.py` | Flower coordinator that sends weights and aggregates bank updates. | **Yes — implement/test** |

### Step 7.1 — Optional: run independent bank training

```powershell
python scripts/training/train_bank.py --bank a
python scripts/training/train_bank.py --bank b
python scripts/training/train_bank.py --bank c
```
---

## 8. Flower: train collaboratively (current development stage)

> **Status: IN PROGRESS.** The original starter ZIP does **not** contain a complete working Flower application. `client_app.py`, `server_app.py`, and `pyproject.toml` must be created, checked against the installed Flower version, and tested. **Do not treat the commands below as proof that a working app already exists.**

### Step 8.1 — Confirm the three required files exist

In the project root:

```powershell
Test-Path .\pyproject.toml
Test-Path .\src\crossbank\fl\client_app.py
Test-Path .\src\crossbank\fl\server_app.py
```

All three must print `True` before trying `flwr run`.

- `pyproject.toml`: registers the application, dependencies, run settings and Flower components.
- `client_app.py`: loads only the appropriate bank CSV, receives global weights, trains, returns updated weights.
- `server_app.py`: initialises the global MLP, dispatches rounds, performs FedAvg and saves outputs.

**Extra caution:** If you copied early draft Flower code from chat, test and adjust it for your **installed Flower version** first. Flower's API/configuration has evolved. For example, ensure the server's `ConfigRecord` and the client's `context.run_config` use consistent keys, and that results are saved to a known working directory.

### Step 8.2 — Install Flower's simulation dependencies

After `pyproject.toml` has been written correctly, from the root with `.venv` active:

```powershell
uv pip install -e ".[simulation]"
```

If your `pyproject.toml` has **no** `[project.optional-dependencies]` entry named `simulation`, use **both** commands instead:

```powershell
uv pip install -e .
uv pip install "flwr[simulation]"
```

Check versions:

```powershell
python -c "import flwr, torch; print('Flower:', flwr.__version__); print('PyTorch:', torch.__version__)"
flwr --help
```

**Windows warning:** Flower local simulation relies on Ray, which can require additional environment setup. If it fails on native Windows, use **WSL2 Ubuntu** for Flower/Ray and, later, Docker/Kubernetes. Do not mix Windows `.venv` executables with a WSL environment; create a fresh Linux `.venv` inside WSL.

### Step 8.3 — Define a 3-bank local simulation

Make sure the `[tool.flwr.app]`, `[tool.flwr.app.components]`, and the local simulation settings in `pyproject.toml` are valid for your installed version. The local federation must have **exactly 3 virtual SuperNodes** for A/B/C.

On current Flower releases, the normal command is:

```powershell
flwr run . --stream
```

This assumes `pyproject.toml` defines a default **local federation with 3 SuperNodes**. An alternative on Flower releases supporting per-run federation overrides is:

```powershell
flwr run . --federation-config "num-supernodes=3 client-resources-num-cpus=1" --stream
```

Use `flwr run --help` if your installed version handles arguments differently.

Official Flower instructions: <https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html> and <https://flower.ai/docs/framework/how-to-run-simulations.html>.

### Step 8.4 — First smoke-test settings

Start small, **not with the whole 6-million-row training set**:

- **3 banks**
- **2 FL rounds**
- **1 local epoch per bank per round**
- **Up to 100,000 transactions per bank**, sampled reproducibly for smoke testing
- **Batch size 1,024**
- **Learning rate 0.001**

With PaySim's severe class imbalance, small random samples can contain very few fraud cases. A smoke test only proves the pipeline runs; it does **not** establish final predictive performance.

### Step 8.5 — What successful FL means

The server initialises global weights → all three banks train their copies → server receives **three weight updates** → FedAvg aggregates them → a new round starts.

**Important distinctions:**

- A Flower **local simulation** may run virtual banks on one computer. That is **not yet** a deployment of three independent networked bank services.
- A saved `global_model.pt` is meaningful only if training actually finished and the file was written successfully. The output path depends on implementation/working directory.
- The previously saved `bank_*_pytorch.pt` files are **not** the inputs to FedAvg.
- A model update being transmitted instead of a CSV is **not** a mathematical privacy guarantee.

---

## 9. Test and compare model performance

We want to answer: **Does collaboration help? What happens when a bank is slow or offline?**

For fair experiments, use a consistent **network architecture, input features, preprocessing, training budget, and chronological validation/test split** wherever possible.

1. **Independent banks:** three separately trained PyTorch models.
2. **Centralised:** one PyTorch MLP trained on all A+B+C data (research upper/reference benchmark, not a privacy-preserving deployment).
3. **Standard FedAvg:** three banks collaboratively train through Flower.
4. **Resilient FedAvg:** proposed improvement that tolerates absent/slow clients.

Measure at least:

- **Fraud precision, recall and F1** (fraud is rare).
- **Average precision / PR-AUC** (more useful than plain accuracy for severe imbalance).
- **Training time per round and total completion time**.
- **Successful aggregation rounds under failures/stragglers**.
- **Model-quality change and delay when 0, 1, or more banks fail**.

Choose prediction thresholds on **validation**, not test data. Use the **test set once** for final comparisons. Record random seed(s), configurations and CPU/GPU environment.

**PaySim caveat:** Very high scores can partly reflect the simulator's account-balance patterns; they are not evidence that the model will work on real financial networks.

---

## 10. Docker and Kubernetes — later steps, NOT finished

**Do not run fictional `docker build` or `kubectl apply` commands yet.** We first need real Dockerfiles, container entrypoints, Flower deployment configuration, Kubernetes manifests, and a tested networking/storage layout.

Planned sequence:

1. Finish and validate local Flower training.
2. Run the Flower deployment components as **separate local processes/services**.
3. Create Docker images for server and bank clients.
4. Set up a local Kubernetes cluster (e.g., Docker Desktop Kubernetes or Minikube).
5. Deploy Flower server and bank services as Kubernetes workloads.
6. Configure data mounts/secrets and avoid putting full PaySim CSVs inside Git or image layers.
7. Deliberately stop/delay a bank and measure recovery/aggregation behaviour.
8. Implement and benchmark our quorum/timeout (and if feasible staleness-aware) aggregation logic.

Kubernetes **restarting a failed container** and the **FL coordinator accepting a quorum without waiting forever** are two different reliability mechanisms. Our research should measure both.

---

## 11. Everyday workflow: coming back tomorrow

Open VS Code → open `crossbank-fl` → open terminal → run:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
Get-ChildItem .\data\processed\*.csv
```

Then choose **one** action:

| You want to... | What to do |
|---|---|
| Check bank data | Run the actual `test_fl_preprocessing.py` path |
| Train bank-only baseline | Run `train_bank.py` (if present and compatible) |
| Work on pooled PyTorch baseline | Create/finish `train_centralized.py` |
| Run real FL logic | Finish/configure Flower then `flwr run . --stream` |
| Work on reliability | Only after baseline Flower rounds work reliably |
| Deploy to Kubernetes | Only after networking/Docker deployment works |

---

## 12. Common problems and fixes

### `ModuleNotFoundError: No module named 'crossbank'`

You may be in the wrong folder or haven't installed your package. From the root:

```powershell
uv pip install -e .
```

Requires a valid `pyproject.toml`. Until then, the old scripts may rely on manual `sys.path` setup.

### `FileNotFoundError: data/processed/bank_a.csv`

Check that your terminal is at the project root and that you've prepared the bank data:

```powershell
pwd
Test-Path .\data\processed\bank_a.csv
```

### `No such file ... scripts/training/train_bank.py`

Your file may still be located directly in `scripts`. Search:

```powershell
Get-ChildItem .\scripts -Recurse -Filter train_bank.py
```

### `flwr is not recognized`

Activate `.venv`; confirm Flower installed:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import flwr; print(flwr.__version__)"
```

### `No pyproject.toml`, `Missing Flower component`, or `No default federation`

The Flower application configuration hasn't been created or completed. Go back to **Section 8**; the old starter archive doesn't include a ready-to-run Flower app.



### `Execution of scripts is disabled on this system`

See the session-only PowerShell execution policy command in Section 2.

---

## 13. Project status checklist

- [x] Downloaded and inspected PaySim (in the development work completed so far)
- [x] Chronological train/validation/test preparation
- [x] Three non-IID bank training partitions
- [x] Early Logistic Regression and XGBoost exploratory benchmarks
- [x] Independent PyTorch bank experiments (older bank-specific scaling)
- [x] Shared nine-feature preprocessing test passed
- [ ] Refactored, fair centralised PyTorch baseline using shared preprocessing
- [ ] Complete, tested Flower `ClientApp` + `ServerApp` + `pyproject.toml`
- [ ] Flower FedAvg smoke test with all 3 banks
- [ ] Global validation/test evaluation and repeatable experiment logs
- [ ] True separated-service Flower deployment
- [ ] Docker container images
- [ ] Kubernetes deployment
- [ ] Client failure/straggler/quorum-aware coordination
- [ ] Final performance and reliability experiments

**What to work on next:** Verify/create `pyproject.toml`, `src/crossbank/fl/client_app.py`, and `src/crossbank/fl/server_app.py`; get **one three-bank Flower FedAvg run** working before changing the cloud infrastructure.

---

## 14. Research and documentation links

- [Flower — PyTorch quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
- [Flower — simulation guide](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [PyTorch](https://pytorch.org/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Kaggle PaySim dataset](https://www.kaggle.com/datasets/moonknightmarvel/paysim)
- [Docker documentation](https://docs.docker.com/)
- [Kubernetes documentation](https://kubernetes.io/docs/home/)

**Security/data note:** Never commit actual sensitive banking data, tokens, passwords, or `.venv` directories. Keep large raw/processed data and generated model checkpoints outside version control when appropriate.
