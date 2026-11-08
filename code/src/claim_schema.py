
from typing import Literal
from pydantic import BaseModel, Field


class ClaimExtraction(BaseModel):
    """Structured information extracted from a damage claim."""

    claim_object: Literal["car", "laptop", "package"]

    issue_types: list[
        Literal[
            "dent",
            "scratch",
            "crack",
            "glass_shatter",
            "broken_part",
            "missing_part",
            "torn_packaging",
            "crushed_packaging",
            "water_damage",
            "stain",
            "none",
            "unknown"
        ]
    ] = Field(
        description="Damage types explicitly claimed by the customer."
    )

    claimed_parts: list[str] = Field(
        description="Object parts the customer actually claims are damaged."
    )

    excluded_parts: list[str] = Field(
        description="Parts the customer explicitly says are not damaged or not included."
    )

    customer_claim_summary: str = Field(
        description="A short summary of the customer's actual damage claim."
    )

    language: str = Field(
        description="Language or languages used by the customer."
    )
