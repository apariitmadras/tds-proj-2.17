import re
from typing import Dict, Any, List, Optional

# --- Helpers to extract questions & expected size ---

def _extract_enumerated_questions(lines: List[str]) -> List[str]:
    """
    Extract sub-questions that start with common enumeration patterns:
      1) question
      1. question
      1- question
      1] question
      1: question
    Returns clean question strings without the leading number token.
    """
    qs: List[str] = []
    pat1 = re.compile(r'^\s*(\d+)\s*[\.\)\]\:]\s*(.+)$')   # 1. / 1) / 1] / 1:
    pat2 = re.compile(r'^\s*(\d+)\s*[-–—]\s*(.+)$')        # 1- / 1– / 1—
    for ln in lines:
        m = pat1.match(ln) or pat2.match(ln)
        if m:
            qs.append(m.group(2).strip())
    return qs

def _extract_expected_len(text: str) -> Optional[int]:
    """
    Detect 'exactly N items' or 'exactly N keys' in the instruction.
    Examples:
      'Return ONLY a JSON array with exactly 4 items'
      'Return a JSON object with exactly 3 keys'
    """
    L = text.lower()
    m = re.search(r'exactly\s+(\d+)\s+(items?|keys?)', L)
    if m:
        try:
            n = int(m.group(1))
            return n if n > 0 else None
        except Exception:
            return None
    return None

# --- Public API ---

def parse_questions_text(qtxt: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "type": "array" | "object",
        "questions": [ ... ],           # may be empty if only length was specified
        "expected_len": Optional[int]   # from 'exactly N items/keys'
      }
    """
    lines = [ln.strip() for ln in qtxt.splitlines() if ln.strip()]
    L = qtxt.lower()

    # Output type (default to array)
    out_type = "object" if ("json object" in L and "json array" not in L) else "array"

    # Enumerated sub-questions
    qs = _extract_enumerated_questions(lines)

    # Expected count if user asked for exact length
    expected_len = _extract_expected_len(L)

    # Fallback: if nothing enumerated and no expected length, treat entire text as one question
    if not qs and expected_len is None:
        qs = [qtxt]

    return {"type": out_type, "questions": qs, "expected_len": expected_len}

def format_answers(out_type: str, questions: List[str], answers: List[Any]) -> Any:
    """
    Enforce exact shape/ordering using the number of questions as the target length.
    (Final length normalization against 'expected_len' is handled in app.py.)
    """
    if out_type == "array":
        return [answers[i] if i < len(answers) else "N/A" for i in range(len(questions))]
    else:
        obj: Dict[str, Any] = {}
        for i, q in enumerate(questions):
            obj[q] = answers[i] if i < len(answers) else "N/A"
        return obj
