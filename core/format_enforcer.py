import re
from typing import Dict, Any, List, Optional

# Heuristics that work even if LLMs are down

def _extract_enumerated_questions(lines: List[str]) -> List[str]:
    qs: List[str] = []
    pat1 = re.compile(r'^\s*(\d+)\s*[\.\)\]\:]\s*(.+)$')  # 1. / 1) / 1] / 1:
    pat2 = re.compile(r'^\s*(\d+)\s*[-–—]\s*(.+)$')       # 1- / 1– / 1—
    for ln in lines:
        m = pat1.match(ln) or pat2.match(ln)
        if m: qs.append(m.group(2).strip())
    return qs

def _extract_expected_len(text: str) -> Optional[int]:
    L = text.lower()
    for pat in [
        r'exactly\s+(\d+)\s+(items?|keys?)',
        r'array\s+of\s+(\d+)',
        r'length\s+(\d+)',
        r'(\d+)\s+items?'
    ]:
        m = re.search(pat, L)
        if m:
            try:
                n = int(m.group(1))
                if n > 0: return n
            except Exception:
                pass
    # fallback: count enumerations
    nums = re.findall(r'^\s*(\d+)\s*[\.\)\]\-:]\s+', text, flags=re.M)
    if nums:
        ints = [int(x) for x in nums]
        return max(ints) if min(ints) == 1 else len(nums)
    return None

def _extract_object_keys(text: str) -> List[str]:
    """
    Try to detect object keys from phrases like:
      - "keys: a, b, c"
      - "with keys [a, b, c]"
      - backticked or quoted lists
    """
    L = text.strip()
    m = re.search(r'keys?\s*[:=]\s*([\[\(]?.+)', L, flags=re.I)
    if m:
        chunk = m.group(1).strip()
        chunk = chunk.splitlines()[0]
        chunk = chunk.strip('[](){}.')
        parts = [p.strip(" `\"'") for p in re.split(r'[,\|/;]', chunk)]
        parts = [p for p in parts if p]
        # de-dup, keep order
        seen, keys = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower()); keys.append(p)
        return keys[:50]
    return []

def parse_request_structure(qtxt: str) -> Dict[str, Any]:
    """
    Returns a *heuristic* structure that we will then refine with the Schema LLM.
    """
    lines = [ln.strip() for ln in qtxt.splitlines() if ln.strip()]
    L = qtxt.lower()

    out_type = "object" if ("json object" in L and "json array" not in L) else "array"
    qs = _extract_enumerated_questions(lines)
    expected_len = _extract_expected_len(qtxt)
    obj_keys: List[str] = _extract_object_keys(qtxt) if out_type == "object" else []

    if out_type == "array":
        if not qs and expected_len is None:
            expected_len = 1
    else:
        if not obj_keys and qs:
            # treat enumerations as keys if object requested
            obj_keys = [q[:80] for q in qs]

    return {
        "out_type": out_type,
        "questions": qs,
        "expected_len": expected_len,
        "obj_keys": obj_keys
    }
