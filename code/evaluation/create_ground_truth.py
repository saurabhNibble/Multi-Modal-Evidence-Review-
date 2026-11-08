
from pathlib import Path
import pandas as pd


# ==========================================
# 1. DEFINE PROJECT PATHS
# ==========================================

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent

SAMPLE_FILE = PROJECT_DIR / "dataset" / "sample_claims.csv"

OUTPUT_FILE = (
    CODE_DIR
    / "evaluation"
    / "extraction_ground_truth.csv"
)


# ==========================================
# 2. LOAD SAMPLE CLAIMS
# ==========================================

df = pd.read_csv(SAMPLE_FILE)

print("Sample claims loaded:", len(df))


# ==========================================
# 3. CREATE GROUND-TRUTH TEMPLATE
# ==========================================

ground_truth = pd.DataFrame({
    "user_id": df["user_id"],
    "claim_object": df["claim_object"],
    "user_claim": df["user_claim"],

    # Fill these manually after reading each conversation
    "expected_issue_types": "",
    "expected_claimed_parts": "",
    "expected_excluded_parts": "",
    "expected_claim_uncertainty": ""
})


# ==========================================
# 4. SAVE TEMPLATE
# ==========================================

ground_truth.to_csv(OUTPUT_FILE, index=False)

print("\nGround-truth template created!")
print("Total rows:", len(ground_truth))
print("Output file:", OUTPUT_FILE)
