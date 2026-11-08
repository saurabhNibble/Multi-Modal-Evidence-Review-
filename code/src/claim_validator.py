
ALLOWED_PARTS = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown"
    },

    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown"
    },

    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown"
    }
}


PART_ALIASES = {
    "car": {
        "front bumper": "front_bumper",
        "rear bumper": "rear_bumper",
        "back bumper": "rear_bumper",
        "side mirror": "side_mirror",
        "wing mirror": "side_mirror",
        "front glass": "windshield",
        "tail light": "taillight",
        "head light": "headlight",
        "car body": "body",
        "body panel": "body",
        "car": "unknown"
    },

    "laptop": {
        "laptop screen": "screen",
        "display": "screen",
        "touchpad": "trackpad",
        "laptop hinge": "hinge",
        "laptop lid": "lid",
        "laptop corner": "corner",
        "usb port": "port",
        "charging port": "port",
        "laptop body": "body",
        "laptop": "unknown"
    },

    "package": {
        "package": "unknown",
        "cardboard box": "box",
        "box corner": "package_corner",
        "package side": "package_side",
        "box side": "package_side",
        "package seal": "seal",
        "shipping label": "label",
        "package contents": "contents",
        "product inside": "item"
    }
}


def normalize_part(part, claim_object):
    """
    Convert a part name into its canonical representation.

    Examples:
        front bumper -> front_bumper
        touchpad -> trackpad
        package -> unknown
    """

    if not isinstance(part, str):
        raise TypeError(f"Part name must be a string, got: {type(part)}")

    part = part.strip().lower()

    if not part:
        return "unknown"

    aliases = PART_ALIASES.get(claim_object, {})

    return aliases.get(part, part)


def validate_claim_parts(extraction):
    """
    Validate claimed and excluded parts against the
    allowed parts for the detected object type.

    Also checks that the same part is not both claimed
    and excluded.
    """

    claim_object = extraction.claim_object

    # Step 1: Validate object type
    if claim_object not in ALLOWED_PARTS:
        raise ValueError(
            f"Unsupported object type: {claim_object}"
        )

    allowed_parts = ALLOWED_PARTS[claim_object]

    # Step 2: Normalize claimed parts
    claimed_parts = {
        normalize_part(part, claim_object)
        for part in extraction.claimed_parts
    }

    # Step 3: Normalize excluded parts
    excluded_parts = {
        normalize_part(part, claim_object)
        for part in extraction.excluded_parts
    }

    # Step 4: Check invalid claimed parts
    invalid_claimed = claimed_parts - allowed_parts

    # Step 5: Check invalid excluded parts
    invalid_excluded = excluded_parts - allowed_parts

    if invalid_claimed or invalid_excluded:
        raise ValueError(
            f"Invalid parts for {claim_object}. "
            f"Invalid claimed parts: {invalid_claimed}. "
            f"Invalid excluded parts: {invalid_excluded}."
        )

    # Step 6: Check overlapping parts
    overlapping_parts = claimed_parts & excluded_parts

    if overlapping_parts:
        raise ValueError(
            f"Parts cannot be both claimed and excluded: "
            f"{overlapping_parts}"
        )

    # Step 7: Save normalized parts back to the Pydantic model
    extraction.claimed_parts = sorted(claimed_parts)
    extraction.excluded_parts = sorted(excluded_parts)

    print("Object-specific part validation successful!")

    return extraction
