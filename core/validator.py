# core/validator.py
"""
Strict validation according to spec + constraints.
- Ensures exact array length or exact object keys (no extras).
- Preserves order where applicable.
- Checks image field shape and optional size limit.
"""
from typing import Any, Dict, List
import base64

class ValidationError(Exception):
    pass

def _is_data_uri_png(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:image/png;base64,")

def _byte_len_from_data_uri(s: str) -> int:
    if not _is_data_uri_png(s):
        return 0
    b64 = s.split(",", 1)[1]
    try:
        return len(base64.b64decode(b64.encode("utf-8"), validate=True))
    except Exception:
        return 0

def validate_output(output: Any, spec: Dict[str, Any], constraints: Dict[str, Any]):
    # Array case
    if spec["kind"] == "array":
        if not isinstance(output, list):
            raise ValidationError("Output must be a JSON array")
        if len(output) != spec["length"]:
            raise ValidationError(f"Array must have exactly {spec['length']} items (got {len(output)})")
        # If image expected, verify at least one slot matches data URI and size
        if constraints.get("needs_image"):
            slot = constraints.get("image_slot")
            idxs = [slot] if isinstance(slot, int) else range(len(output))
            found = False
            for i in idxs:
                if isinstance(i, int) and 0 <= i < len(output) and _is_data_uri_png(output[i]):
                    found = True
                    if "image_size_max" in constraints:
                        size = _byte_len_from_data_uri(output[i])
                        if size > constraints["image_size_max"]:
                            raise ValidationError(f"Image size {size} exceeds limit {constraints['image_size_max']} bytes")
                    break
            if not found:
                last = output[-1]
                if not _is_data_uri_png(last):
                    raise ValidationError("Expected a base64 PNG data URI in the designated slot (or last item)")
                if "image_size_max" in constraints:
                    size = _byte_len_from_data_uri(last)
                    if size > constraints["image_size_max"]:
                        raise ValidationError(f"Image size {size} exceeds limit {constraints['image_size_max']} bytes")
        return

    # Object case
    if spec["kind"] == "object":
        if not isinstance(output, dict):
            raise ValidationError("Output must be a JSON object")
        keys = spec["keys"]
        if set(output.keys()) != set(keys):
            raise ValidationError(f"Object must contain exactly keys {keys} (no extras/missing)")
        if list(output.keys()) != keys:
            raise ValidationError("Object keys must be in the requested order")
        if constraints.get("needs_image"):
            slot = constraints.get("image_slot")
            target_key = slot if isinstance(slot, str) else "image"
            if target_key not in output or not _is_data_uri_png(output[target_key]):
                raise ValidationError(f'Expected key "{target_key}" to be a base64 PNG data URI')
            if "image_size_max" in constraints:
                size = _byte_len_from_data_uri(output[target_key])
                if size > constraints["image_size_max"]:
                    raise ValidationError(f"Image size {size} exceeds limit {constraints['image_size_max']} bytes")
        return

    raise ValidationError("Unknown spec kind")
