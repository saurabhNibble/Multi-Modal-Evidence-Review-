
import base64
import json
import urllib.request
import urllib.error
from pathlib import Path

image_path = Path(r"test_gemma_input.png")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Read and encode image
image_bytes = image_path.read_bytes()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

print("Image size:", len(image_bytes), "bytes")
print("Sending image to Gemma 4...")

# Use Ollama Chat API
payload = {
    "model": "gemma4:e2b",
    "messages": [
        {
            "role": "user",
            "content": (
                "Inspect this damage claim photograph. "
                "Describe the visible object, damaged part, "
                "damage type, image quality, and anything "
                "that cannot be determined. "
                "Do not assume damage that is not visible."
            ),
            "images": [image_base64]
        }
    ],
    "stream": False
}

request = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))

    print("\nGemma 4 Vision Response:\n")
    print(result["message"]["content"])

except urllib.error.HTTPError as error:
    print("\nHTTP Error:", error.code)
    print("Ollama response:")
    print(error.read().decode("utf-8"))

except Exception as error:
    print("\nUnexpected error:", error)
