from pathlib import Path
from src.data_loader import load_dataset
from PIL import Image
import cv2


BASE_DIR = Path(__file__).resolve().parents[1]


def main():
    print("="*50)
    print("Multi-Modal Claim Verification System")
    print("="*50)

    print("Project Root Directory:", BASE_DIR)
    # Load datasets
    datasets = load_dataset()


    #get claims dataset
    claims_df = datasets["claims"]

    #select the first customer claim
    claim = claims_df.iloc[0]

    print("\n Customer Details System")
    print("-"*30)

    print("User ID:",claim["user_id"])
    print("Claim Object:", claim["claim_object"])
    print("User Claim:", claim["user_claim"])

    # Get uploaded image paths

    image_paths = claim["image_paths"].split(";")

    print("\n Uploaded Images:")

    for image_path in image_paths:

        full_path = BASE_DIR / "dataset" / image_path.strip()

        if full_path.exists():
            print("Image found:", full_path)

            #open the image
            image = Image.open(full_path)

            #display the image
            image.show()
            print("Image size:", image.size) # has list of (width, height)
            height, width = image.size
            if width < 500 or height < 500:
                print("Warning: Low-resolution image")
            else:
                print("Image resolution looks acceptable")
        else:
            print("Image not found:", full_path)

if __name__ == "__main__":
    main()

#if __name__ == "__main__": means: Run this function only when I execute this file directly, not when another file imports it.