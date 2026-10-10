
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from crossbank.data.pipeline import partition_non_iid

# 1. Load the original PaySim dataset
src = Path("data/raw/paysim.csv")

if not src.exists():
    raise FileNotFoundError(f"Missing dataset: {src}")

print("Loading PaySim...")
df = pd.read_csv(src)

# 2. Same chronological split as XGBoost
train = df[df["step"] <= 520].copy()

validation = df[
    (df["step"] > 520) & (df["step"] <= 632)
].copy()

test = df[df["step"] > 632].copy()

print(f"Training: {len(train):,}")
print(f"Validation: {len(validation):,}")
print(f"Testing: {len(test):,}")

# 3. Split training data into 3 simulated banks
banks = partition_non_iid(train)

# 4. Save each bank's dataset
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

for i, part in enumerate(banks):
    bank_name = chr(65 + i)

    filename = out / f"bank_{bank_name.lower()}.csv"

    part.to_csv(filename, index=False)

    print(
        f"Bank {bank_name}: "
        f"{len(part):,} transactions, "
        f"{int(part['isFraud'].sum()):,} fraud cases, "
        f"fraud rate: {part['isFraud'].mean():.4%}"
    )

# 5. Save validation and test datasets
validation.to_csv(out / "validation.csv", index=False)
test.to_csv(out / "test.csv", index=False)

print("\nBank datasets prepared successfully!")
