
import kagglehub
from pathlib import Path
import shutil

# Download the PaySim dataset from Kaggle
download_path = kagglehub.dataset_download(
    "moonknightmarvel/paysim"
)

print("Downloaded to:", download_path)

# Find the CSV file
csv_files = list(Path(download_path).rglob("paysim.csv"))

if not csv_files:
    raise FileNotFoundError(
        "paysim.csv not found in downloaded dataset"
    )

# Copy into our project's raw data folder
destination = Path("data/raw/paysim.csv")
destination.parent.mkdir(parents=True, exist_ok=True)

shutil.copy2(csv_files[0], destination)

print("Dataset saved to:", destination)
print("Size:", round(destination.stat().st_size / 1e6, 2), "MB")
