
from pathlib import Path
import pandas as pd


# Find the project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset folder location
DATASET_DIR = BASE_DIR / "dataset"


def load_dataset():

    print("\nLoading datasets...")

    files = {
        "sample_claims": "sample_claims.csv",
        "claims": "claims.csv",
        "user_history": "user_history.csv",
        "evidence_requirements": "evidence_requirements.csv"
    }

    datasets = {}

    for name, filename in files.items():

        file_path = DATASET_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        datasets[name] = pd.read_csv(file_path)

        print(f"\n{name}")
        print("Rows:", len(datasets[name]))
        print("Columns:", datasets[name].columns.tolist())

    print("\nAll datasets loaded successfully!")

    return datasets
