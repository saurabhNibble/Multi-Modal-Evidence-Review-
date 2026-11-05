
from pathlib import Path
from src.data_loader import load_dataset
from PIL import Image
import cv2
import numpy as np


# Project root directory
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

    # ==========================================
    # INSPECT EVIDENCE REQUIREMENTS
    # ==========================================

    requirements_df = datasets["evidence_requirements"]

    print("\nEVIDENCE REQUIREMENTS")
    print("-" * 50)

    # Display the first 11 rows of the evidence requirements DataFrame
    # Convert the DataFrame into a readable string
    # Hide the row index while printing

    print(requirements_df.head(11).to_string(index=False))
    

    # Get claims dataset
    claims_df = datasets["claims"]

    # Select the first customer claim
    claim = claims_df.iloc[0]

    # Get the customer's claim object
    claim_object = claim["claim_object"]

    # Filter evidence requirements for this object
    object_requirements = requirements_df[
        requirements_df["claim_object"] == claim_object
    ]

    print("\nREQUIREMENTS FOR THIS CLAIM OBJECT")
    print(object_requirements.to_string(index=False))

    # ==========================================
    # 2. DISPLAY CUSTOMER CLAIM DETAILS
    # ==========================================

    print("\nCUSTOMER CLAIM DETAILS")
    print("-" * 30)

    print("User ID:", claim["user_id"])
    print("Claim Object:", claim["claim_object"])
    print("User Claim:", claim["user_claim"])

    # ==========================================
    # 3. GET UPLOADED IMAGE PATHS
    # ==========================================

    image_paths = str(claim["image_paths"]).split(";")

    print("\nUPLOADED IMAGES")
    print("-" * 30)

    # ==========================================
    # 4. PROCESS EACH IMAGE
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
            image = Image.open(full_path)

            print("Image Format:", image.format)
            print("Image Size:", image.size)

            # ==========================================
            # 5. CHECK IMAGE RESOLUTION
            # ==========================================

            width, height = image.size

            print("Width:", width)
            print("Height:", height)

            if width < 500 or height < 500:
                print("Warning: Low-resolution image")
            else:
                print("Image resolution looks acceptable")

            # ==========================================
            # 6. DETECT BLUR
            # ==========================================

            # Convert image into grayscale NumPy array
            gray = np.array(image.convert("L"))

            # Calculate Laplacian variance
            blur_score = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

            print("Blur Score:", round(blur_score, 2))

            # Basic blur classification
            if blur_score < 100:
                print("Warning: Image may be blurry")
            else:
                print("Image appears sharp")

        except Exception as e:

            print("Error processing image:", e)

    print("\n" + "=" * 50)
    print("IMAGE PROCESSING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
