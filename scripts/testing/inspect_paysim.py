
import pandas as pd

path = "data/raw/paysim.csv"

# Read only what we need, in chunks
for chunk in pd.read_csv(path, chunksize=100_000):
    fraud = chunk[chunk["isFraud"] == 1]

    if not fraud.empty:
        print("Fraudulent transactions:")
        print(fraud.head(10).to_string(index=False))
        break

#----------------- Checking number of flagged rows --------------------
total_above = 0
flagged_above = 0
unflagged_above = 0

for chunk in pd.read_csv(
    path,
    usecols=["type", "amount", "isFraud", "isFlaggedFraud"],
    chunksize=100_000,
):
    above = (
        (chunk["type"] == "TRANSFER")
        & (chunk["amount"] > 200_000)
    )

    total_above += above.sum()
    flagged_above += (
        above & (chunk["isFlaggedFraud"] == 1)
    ).sum()
    unflagged_above += (
        above & (chunk["isFlaggedFraud"] == 0)
    ).sum()

print("Transfers above 200,000:", total_above)
print("Flagged:", flagged_above)
print("Not flagged:", unflagged_above)

#----------------- Flagged Fraud that is 200k and above, it only flags transfer --------------------
flagged_rows = []

for chunk in pd.read_csv(path, chunksize=100_000):
    flagged = chunk[chunk["isFlaggedFraud"] == 1]

    if not flagged.empty:
        flagged_rows.append(flagged)

if flagged_rows:
    result = pd.concat(flagged_rows)

    # Print all flagged transactions
    print("\n=== FLAGGED TRANSACTIONS ===\n")
    print(result.to_string(index=False))

    print("\nTotal flagged:", len(result))

    # Check whether they were actually fraudulent
    print("\nActual fraud labels:")
    print(result["isFraud"].value_counts())

    # Check transaction amounts
    print("\nTransaction amounts:")
    print(result[["type", "amount", "isFraud", "isFlaggedFraud"]])

else:
    print("No flagged transactions found.")
