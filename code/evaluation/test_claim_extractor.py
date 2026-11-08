
import sys
from pathlib import Path
import pandas as pd

# ==========================================
# 1. SET PROJECT PATHS
# ==========================================

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent

sys.path.insert(0, str(CODE_DIR))

from code.src.llm_claim_extractor import extract_claim


# ==========================================
# 2. LOAD SAMPLE DATASET
# ==========================================

sample_path = PROJECT_DIR / "dataset" / "sample_claims.csv"

df = pd.read_csv(sample_path)

print("\nTOTAL SAMPLE CLAIMS:", len(df))

# Check duplicate user IDs
duplicate_ids = df[df["user_id"].duplicated(keep=False)]

if not duplicate_ids.empty:
    print("\nWARNING: Duplicate user IDs found!")
    print(duplicate_ids["user_id"].tolist())
else:
    print("No duplicate user IDs found.")


# ==========================================
# 3. SELECT CLAIMS FOR TESTING
# ==========================================

test_df = df.copy()

print("\nCLAIMS SELECTED FOR TESTING:", len(test_df))


# ==========================================
# 4. INITIALIZE RESULTS
# ==========================================

results = []


# ==========================================
# 5. TEST EACH CLAIM
# ==========================================

for index, row in test_df.iterrows():

    user_id = row["user_id"]

    print("\n" + "=" * 60)
    print(f"Testing row: {index}")
    print(f"User ID: {user_id}")
    print("=" * 60)

    try:

        prediction = extract_claim(
            conversation=row["user_claim"],
            claim_object=row["claim_object"]
        )

        predicted_issues = prediction.issue_types
        predicted_parts = prediction.claimed_parts

        expected_issue = str(row["issue_type"]).strip()
        expected_part = str(row["object_part"]).strip()

        issue_match = expected_issue in predicted_issues
        part_match = expected_part in predicted_parts

        results.append({

            "row_index": index,
            "user_id": user_id,

            "expected_issue": expected_issue,
            "predicted_issue": ";".join(predicted_issues),
            "issue_match": issue_match,

            "expected_part": expected_part,
            "predicted_part": ";".join(predicted_parts),
            "part_match": part_match,

            "extraction_success": True,
            "error": ""

        })

    except Exception as error:

        print("\nEXTRACTION FAILED!")
        print("Error:", error)

        results.append({

            "row_index": index,
            "user_id": user_id,

            "expected_issue": str(row["issue_type"]),
            "predicted_issue": "FAILED",
            "issue_match": False,

            "expected_part": str(row["object_part"]),
            "predicted_part": "FAILED",
            "part_match": False,

            "extraction_success": False,
            "error": str(error)

        })


# ==========================================
# 6. CREATE RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(results)


# ==========================================
# 7. DISPLAY EXTRACTION COMPARISON
# ==========================================

print("\n\nEXTRACTION COMPARISON")
print("=" * 80)

print(results_df.to_string(index=False))


# ==========================================
# 8. CALCULATE EVALUATION METRICS
# ==========================================

total_claims = len(results_df)

successful_claims = int(
    results_df["extraction_success"].sum()
)

failed_claims = total_claims - successful_claims

issue_matches = int(results_df["issue_match"].sum())
part_matches = int(results_df["part_match"].sum())

if successful_claims > 0:

    issue_accuracy = issue_matches / successful_claims * 100
    part_accuracy = part_matches / successful_claims * 100

else:

    issue_accuracy = 0
    part_accuracy = 0

validation_success_rate = (
    successful_claims / total_claims * 100
    if total_claims else 0
)


# ==========================================
# 9. PRINT SUMMARY
# ==========================================

print("\n\nEVALUATION SUMMARY")
print("=" * 60)

print("Total claims tested:", total_claims)
print("Successful extractions:", successful_claims)
print("Failed extractions:", failed_claims)

print(
    "Validation success rate:",
    f"{validation_success_rate:.2f}%"
)

print("Issue matches:", issue_matches, "/", successful_claims)
print("Issue accuracy:", f"{issue_accuracy:.2f}%")

print("Part matches:", part_matches, "/", successful_claims)
print("Part accuracy:", f"{part_accuracy:.2f}%")


# ==========================================
# 10. DISPLAY ORIGINAL MISMATCHED CLAIMS
# ==========================================

print("\n\nMISMATCHED CLAIM CONVERSATIONS")
print("=" * 80)

mismatches = results_df[
    results_df["extraction_success"]
    & (
        (~results_df["issue_match"])
        | (~results_df["part_match"])
    )
]

if mismatches.empty:

    print("No mismatches found!")

else:

    for _, mismatch in mismatches.iterrows():

        # Use row_index instead of user_id.
        # This ensures we retrieve the exact original row.

        original_row = df.loc[int(mismatch["row_index"])]

        print("\nRow index:", mismatch["row_index"])
        print("User ID:", mismatch["user_id"])
        print("Claim object:", original_row["claim_object"])

        print("\nOriginal conversation:")
        print(original_row["user_claim"])

        print("\nExpected issue:", mismatch["expected_issue"])
        print("Predicted issue:", mismatch["predicted_issue"])

        print("\nExpected part:", mismatch["expected_part"])
        print("Predicted part:", mismatch["predicted_part"])

        print("-" * 80)


# ==========================================
# 11. DISPLAY EXTRACTION FAILURES
# ==========================================

print("\n\nEXTRACTION FAILURES")
print("=" * 80)

failures = results_df[
    ~results_df["extraction_success"]
]

if failures.empty:

    print("No extraction failures!")

else:

    for _, failure in failures.iterrows():

        print("\nRow index:", failure["row_index"])
        print("User ID:", failure["user_id"])
        print("Error:", failure["error"])


# ==========================================
# 12. SAVE EVALUATION RESULTS
# ==========================================

output_path = (
    CODE_DIR
    / "evaluation"
    / "claim_extraction_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\nEvaluation results saved to:")
print(output_path)

print("\nCLAIM EXTRACTION EVALUATION COMPLETED!")
