import json
from typing import Any, Dict, List

def _coerce(hint: str, v: Any):
    if hint == "int":
        try: return int(v)
        except Exception: return 0
    if hint == "float":
        try: return float(v)
        except Exception: return 0.0
    if hint == "bool":
        return bool(v)
    # others are string-ish
    if isinstance(v, (dict, list)): return json.dumps(v, ensure_ascii=False)
    return str(v) if v is not None else ""

def enforce_shape_and_types(schema: Dict[str, Any], values: Any) -> Any:
    """
    Ensure exact shape and basic type coercion by the hints.
    """
    hints = schema.get("hints") or []
    if schema["out_type"] == "array":
        n = max(1, int(schema.get("target_len", 1)))
        arr = list(values) if isinstance(values, list) else []
        # pad or trim
        if len(arr) < n: arr += [""] * (n - len(arr))
        if len(arr) > n: arr = arr[:n]
        # coerce
        out = []
        for i in range(n):
            hint = hints[i] if i < len(hints) else "string"
            out.append(_coerce(hint, arr[i]))
        return out
    else:
        keys = schema.get("keys") or ["answer"]
        obj = values if isinstance(values, dict) else {}
        out = {}
        for i, k in enumerate(keys):
            v = obj.get(k, "")
            hint = hints[i] if i < len(hints) else "string"
            out[k] = _coerce(hint, v)
        # preserve key order
        return {k: out[k] for k in keys}
