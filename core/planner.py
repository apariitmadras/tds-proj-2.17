# core/planner.py
"""
Planner module: builds a formal spec of the required output strictly from `questions.txt`.
- If OPENAI_API_KEY_PLANNER is present, you could call an LLM here (placeholder).
- Otherwise, use robust heuristics (regex-based) to detect array/object shape and constraints.
"""
import re
from typing import Dict, List, Any

ARRAY_RE = re.compile(r'JSON\s+array.*?exactly\s+(\d+)\s+item', re.I | re.S)
OBJECT_KEYS_RE = re.compile(r'JSON\s+object.*?keys?\s*:?(.+)', re.I)
KEY_SPLIT_RE = re.compile(r'[,\n]+')

def _heuristic_spec(text: str) -> Dict[str, Any]:
    # Array with exact N items?
    m_arr = ARRAY_RE.search(text)
    if m_arr:
        n = int(m_arr.group(1))
        return {
            "kind": "array",
            "length": n,
            "types": ["string"] * n,
        }
    # Object with specific keys?
    m_obj = OBJECT_KEYS_RE.search(text)
    if m_obj:
        keys = [k.strip() for k in KEY_SPLIT_RE.split(m_obj.group(1)) if k.strip()]
        return {
            "kind": "object",
            "keys": keys,
            "types": {k: "string" for k in keys},
            "order": keys,
        }
    # Fallback
    return {
        "kind": "array",
        "length": 1,
        "types": ["string"],
    }

def plan_spec(text: str) -> Dict[str, Any]:
    # TODO: Optional LLM call using OPENAI_API_KEY_PLANNER and PLANNER_MODEL
    return _heuristic_spec(text)
