# utils/format_detect.py
import re
from typing import Dict, Any

def wants_image(text: str) -> bool:
    return bool(re.search(r'base-?64.*image|data:image/png;base64', text, re.I))

def parse_user_constraints(text: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    constraints["needs_image"] = wants_image(text)
    m = re.search(r'under\s+([\d,]+)\s*bytes', text, re.I)
    if m:
        try:
            constraints["image_size_max"] = int(m.group(1).replace(",", ""))
        except Exception:
            pass
    if re.search(r'last\s+item', text, re.I):
        constraints["image_slot"] = None  # assembler puts image at last
    return constraints
