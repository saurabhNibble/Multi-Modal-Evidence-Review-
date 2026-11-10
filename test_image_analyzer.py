
from code.src.image_analyzer import analyze_image

image_path = r".\dataset\images\test\case_001\img_1.jpg"

try:
    result = analyze_image(image_path)

    print("\n===== GEMMA IMAGE ANALYSIS =====")

    for key, value in result.items():
        print(f"\n{key}: {value}")

except Exception as error:
    print("Image analysis failed:", error)
