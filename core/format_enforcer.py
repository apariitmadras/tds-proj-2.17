import re, json
from typing import Dict, Any, List

def parse_questions_text(qtxt: str) -> Dict[str, Any]:
    lines = [ln.strip() for ln in qtxt.splitlines() if ln.strip()]
    # Detect requested output type
    L = qtxt.lower()
    if "json object" in L and "json array" not in L:
        out_type = "object"
    elif "json array" in L and "json object" not in L:
        out_type = "array"
    else:
        # default to array
        out_type = "array"

    # Extract numbered questions "1. ...."
    qs: List[str] = []
    for ln in lines:
        m = re.match(r"^\s*(\d+)\.(.*)$", ln)
        if m:
            qs.append(m.group(2).strip())
    if not qs:
        qs = [qtxt]
    return {"type": out_type, "questions": qs}

def format_answers(out_type: str, questions: List[str], answers: List[Any]) -> Any:
    # Ensure exact shape/order. Values become strings unless already JSON-y.
    if out_type == "array":
        # ensure array length equals number of questions
        arr: List[Any] = []
        for i in range(len(questions)):
            v = answers[i] if i < len(answers) else "N/A"
            arr.append(v)
        return arr
    else:
        obj: Dict[str, Any] = {}
        for i, q in enumerate(questions):
            v = answers[i] if i < len(answers) else "N/A"
            obj[q] = v
        return obj
