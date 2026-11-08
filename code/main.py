
from pathlib import Path

from src.data_loader import load_dataset
from src.claim_extractor import extract_issue_types, extract_object_parts

from PIL import Image
import cv2
import numpy as np


# ==========================================
# PROJECT ROOT DIRECTORY
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]


def main():

    print("=" * 50)
    print("MULTI-MODAL CLAIM VERIFICATION SYSTEM")
    print("=" * 50)

    print("Project Root Directory:", BASE_DIR)

    # ==========================================
    # 1. LOAD DATASETS
    # ==========================================

    datasets = load_dataset()

    # Get evidence requirements dataset
    requirements_df = datasets["evidence_requirements"]

    print("\nEVIDENCE REQUIREMENTS")
    print("-" * 50)

    print(requirements_df.to_string(index=False))

    # Get customer claims dataset
    claims_df = datasets["claims"]

    # Count total claims
    total_claims = len(claims_df)

    print("\nTotal Claims to Process:", total_claims)

    # ==========================================
    # 2. PROCESS ALL CUSTOMER CLAIMS
    # ==========================================

    for index, claim in claims_df.iterrows():

        print("\n" + "=" * 50)
        print(f"PROCESSING CLAIM {index + 1} / {total_claims}")
        print("=" * 50)

        # ==========================================
        # 3. DISPLAY CUSTOMER CLAIM DETAILS
        # ==========================================

        user_id = claim["user_id"]
        claim_object = claim["claim_object"]
        user_claim = claim["user_claim"]

        print("\nCUSTOMER CLAIM DETAILS")
        print("-" * 40)

        print("User ID:", user_id)
        print("Claim Object:", claim_object)
        print("User Claim:", user_claim)

        # ==========================================
        # 4. CLAIM EXTRACTION
        # ==========================================

        detected_issues = extract_issue_types(user_claim)

        detected_parts = extract_object_parts(
        user_claim,
        claim_object
        )

        print("\nCLAIM EXTRACTION RESULTS")
        print("-" * 40)

        print("Detected Issues:", detected_issues)
        print("Detected Parts:", detected_parts)

        # ==========================================
        # 5. FILTER EVIDENCE REQUIREMENTS
        # ==========================================

        # Get requirements for the current object.
        # Also include general requirements that apply to all objects.

        object_requirements = requirements_df[
            requirements_df["claim_object"].isin(
                ["all", claim_object]
            )
        ]

        print("\nREQUIREMENTS FOR THIS CLAIM OBJECT")
        print("-" * 40)

        print(object_requirements.to_string(index=False))

        print("\nAVAILABLE EVIDENCE CATEGORIES")

        for requirement in object_requirements["applies_to"]:
            print("-", requirement)

        # ==========================================
        # 6. GET UPLOADED IMAGE PATHS
        # ==========================================

        image_paths_value = claim["image_paths"]

        print("\nUPLOADED IMAGES")
        print("-" * 40)

        if not isinstance(image_paths_value, str):
            print("No valid image paths found.")
            continue

        image_paths = image_paths_value.split(";")

        # ==========================================
        # 7. PROCESS EACH IMAGE
        # ==========================================

        for image_path in image_paths:

            image_path = image_path.strip()

            if not image_path:
                continue

            full_path = BASE_DIR / "dataset" / image_path

            if not full_path.exists():

                print("\nImage not found:", full_path)
                continue

            print("\nImage found:", full_path)

            try:

                # Open image using Pillow
                with Image.open(full_path) as image:

                    print("Image Format:", image.format)
                    print("Image Size:", image.size)

                    # ==========================================
                    # 8. CHECK IMAGE RESOLUTION
                    # ==========================================

                    width, height = image.size

                    print("Width:", width)
                    print("Height:", height)

                    if width < 500 or height < 500:
                        print("Warning: Low-resolution image")
                    else:
                        print("Image resolution looks acceptable")

                    # ==========================================
                    # 9. DETECT IMAGE BLUR
                    # ==========================================

                    # Convert image to grayscale
                    gray = np.array(image.convert("L"))

                    # Calculate Laplacian variance
                    blur_score = cv2.Laplacian(
                        gray,
                        cv2.CV_64F
                    ).var()

                    print("Blur Score:", round(blur_score, 2))

                    # Initial blur classification
                    if blur_score < 100:
                        print("Warning: Image may be blurry")
                    else:
                        print("Image appears sharp")

            except Exception as e:

                print("Error processing image:", e)

    # ==========================================
    # 10. PROCESSING COMPLETED
    # ==========================================

    print("\n" + "=" * 50)
    print("ALL CLAIMS PROCESSED SUCCESSFULLY")
    print("Total Claims:", total_claims)
    print("=" * 50)


if __name__ == "__main__":
    main()
