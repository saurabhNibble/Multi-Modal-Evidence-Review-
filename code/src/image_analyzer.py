
import base64
import io
import json
import urllib.request
import urllib.error

from PIL import Image


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma4:e2b"


def analyze_image(image_path: str) -> dict:
    """
    Analyze one damage-claim image using local Gemma 4 Vision.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary containing the visual findings.
    """

    # 1. Open the image and convert it to genuine PNG format.
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")

        image_bytes = image_buffer.getvalue()

    # 2. Encode the PNG image as Base64.
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # 3. Ask Gemma to return structured visual findings.
    prompt = """
You are a visual damage evidence analyst.

Inspect the attached photograph carefully.

Return ONLY a valid JSON object with these fields:

{
    "visible_object": "object identified in the image",
    "visible_parts": ["parts clearly visible"],
    "damaged_parts": ["parts with clearly visible damage"],
    "damage_types": ["visible damage types"],
    "damage_description": "brief description of visible evidence",
    "damage_visibility": "visible, not_visible, or unclear",
    "image_quality": "good, poor, or unusable",
    "uncertainties": ["things that cannot be determined from this image"]
}

Important rules:
- The photograph is the primary source of evidence.
- Do not assume damage exists just because the customer claims it.
- Do not invent damage that is not visually observable.
- Do not treat shadows, reflections, dirt, or normal design features
  as damage without clear evidence.
- Report only parts visible in this photograph.
- If damage cannot be confirmed, use an empty damaged_parts list.
- If the image is unclear, explain why in uncertainties.
- Do not make a final insurance claim decision.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    # 4. Send the request to Ollama.
    request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            response_data = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        error_message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP error {error.code}: {error_message}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama: {error.reason}"
        ) from error

    # 5. Extract and parse Gemma's JSON response.
    try:
        content = response_data["message"]["content"]
        findings = json.loads(content)

    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Gemma returned an invalid JSON response: {response_data}"
        ) from error

    # 6. Check that the expected fields are present.
    required_fields = [
        "visible_object",
        "visible_parts",
        "damaged_parts",
        "damage_types",
        "damage_description",
        "damage_visibility",
        "image_quality",
        "uncertainties"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in findings
    ]

    if missing_fields:
        raise ValueError(
            f"Gemma response is missing fields: {missing_fields}"
        )

    return findings
