import os, hashlib, base64
from typing import List, Any

MODE = os.getenv("FALLBACK_MODE", "placeholders")  # placeholders | synthetic

def _synthetic(i: int, seed: str) -> str:
    h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
    token = base64.urlsafe_b64encode(h[:9]).decode().strip("=")
    return f"~synthetic:{token}~"

def make_fallback_answers(n: int, seed: str = "default") -> List[Any]:
    if MODE == "synthetic":
        return [_synthetic(i, seed) for i in range(n)]
    return ["N/A" for _ in range(n)]
