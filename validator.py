# validator.py — drop-in
import json
import re
from typing import Any, Dict, List

_DATA_URI_FALLBACK = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHYALtTyR2wQAAAABJRU5ErkJggg=="

def _coerce(hint: str, v: Any):
    h = (hint or "string").lower()

    # Numeric-like
    if h in ("int", "year"):
        try:
            return int(v)
        except Exception:
            return 0
    if h in ("float", "corr"):
        try:
            return float(v)
        except Exception:
            return 0.0

    # Boolean
    if h == "bool":
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    # Date (YYYY-MM-DD)
    if h == "date":
        s = "" if v is None else str(v)
        return s if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) else "2000-01-01"

    # URL
    if h == "url":
        s = "" if v is None else str(v)
        return s if s.startswith(("http://", "https://")) else "https://example.com/"

    # PNG data URI
    if h == "png":
        s = "" if v is None else str(v)
        return s if s.startswith("data:image/png;base64,") else _DATA_URI_FALLBACK

    # Titles / names / generic string
    if h in ("title", "name", "string"):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return "" if v is None else str(v)

    # Default: stringify safely
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v)

def enforce_shape_and_types(schema: Dict[str, Any], values: Any) -> Any:
    """
    Ensure exact shape and basic type coercion by the per-item hints.
    """
    hints: List[str] = list(schema.get("hints") or [])

    if schema.get("out_type") == "array":
        n = max(1, int(schema.get("target_len", 1)))
        arr = list(values) if isinstance(values, list) else []
        # pad / trim
        if len(arr) < n:
            arr += [""] * (n - len(arr))
        elif len(arr) > n:
            arr = arr[:n]
        # align hints
        if len(hints) < n:
            hints += ["string"] * (n - len(hints))
        # coerce
        out = []
        for i in range(n):
            out.append(_coerce(hints[i], arr[i]))
        return out

    # object
    keys: List[str] = list(schema.get("keys") or ["answer"])
    obj = values if isinstance(values, dict) else {}
    if len(hints) < len(keys):
        hints += ["string"] * (len(keys) - len(hints))
    coerced = {}
    for i, k in enumerate(keys):
        coerced[k] = _coerce(hints[i], obj.get(k, ""))
    # preserve key order
    return {k: coerced[k] for k in keys}
