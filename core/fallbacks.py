import os, random, string
from typing import Any, List

FALLBACK_MODE = os.getenv("FALLBACK_MODE", "placeholders")  # 'placeholders' or 'synthetic'
FAKE_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]

def placeholder_value() -> str:
    # Safe placeholder that is clearly not factual
    return "N/A"

def synthetic_value() -> str:
    # Non-factual synthetic (seeded gibberish) to keep format valid without claiming truth
    token = "-".join(random.sample(FAKE_WORDS, k=2))
    # Add a random suffix to avoid looking authoritative
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"~synthetic:{token}:{suffix}~"

def make_fallback_answers(n: int) -> List[Any]:
    out = []
    for _ in range(n):
        if FALLBACK_MODE == "synthetic":
            out.append(synthetic_value())
        else:
            out.append(placeholder_value())
    return out
