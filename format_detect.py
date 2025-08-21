import re, json, asyncio
from typing import Any, Dict, List, Optional
from llm import ask

# Heuristic helpers
ENUM_RX = re.compile(r'^\s*(\d+)\s*[\.\)\]\-:]\s*(.+)$', re.M)

def _heuristic_schema(text: str) -> Dict[str, Any]:
    L = (text or "").lower()
    out_type = "object" if ("json object" in L and "json array" not in L) else "array"

    lines = ENUM_RX.findall(text or "")  # list of (num, content)
    questions = [c.strip() for _, c in lines]
    keys: List[str] = []
    target_len: int = 0

    if out_type == "array":
        # exact length?
        m = re.search(r'exactly\s+(\d+)\s+(?:items?|keys?)', L)
        if m:
            target_len = int(m.group(1))
        elif questions:
            nums = [int(n) for n,_ in lines]
            target_len = max(nums) if (min(nums)==1) else len(lines)
        else:
            target_len = 1
    else:
        m = re.search(r'keys?\s*:\s*([^\n]+)', text, flags=re.I)
        if m:
            keys = [k.strip(" `\"'") for k in m.group(1).split(",") if k.strip()]
        else:
            keys = [q[:64] for q in questions] or ["answer"]
        target_len = len(keys)

    # crude hints
    def infer_hint(s: str) -> str:
        sL = s.lower()
        if re.search(r'\b(integer|int)\b', sL): return "int"
        if re.search(r'\b(float|double|decimal|number|numeric)\b', sL): return "float"
        if re.search(r'\b(boolean|bool)\b', sL): return "bool"
        if "correlation" in sL: return "corr"
        if "png" in sL and "base64" in sL: return "png"
        if any(w in sL for w in ["scatterplot","plot","chart","graph"]): return "png"
        if "year" in sL: return "year"
        if "date" in sL: return "date"
        if "url" in sL or "link" in sL: return "url"
        if "title" in sL: return "title"
        if "name" in sL: return "name"
        if any(w in sL for w in ["how many","count","number of","total"]): return "int"
        return "string"

    hints = []
    items = (questions if out_type=="array" else keys)
    for it in items:
        hints.append(infer_hint(it))

    return {"out_type": out_type, "target_len": target_len, "keys": keys, "hints": hints, "questions": questions}

async def detect_schema(instruction_text: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Try LLM schema extractor; on failure, fallback to heuristics.
    """
    user = instruction_text.strip()
    # Try LLM first (if configured)
    sys_prompt = open("prompts/schema_extractor.md","r",encoding="utf-8").read()
    res = await ask("schema", sys_prompt, user, timeout=timeout)
    if res:
        try:
            data = json.loads(res)
            # minimal sanity
            if "out_type" in data and (data.get("out_type") in ("array","object")):
                # fill deriveds
                if data["out_type"]=="array":
                    data.setdefault("target_len", max(1, len(data.get("hints", [])) or len(data.get("questions", [])) or 1))
                    data.setdefault("keys", [])
                else:
                    ks = data.get("keys") or []
                    data["keys"] = ks
                    data["target_len"] = len(ks)
                data.setdefault("hints", [])
                data.setdefault("questions", [])
                return data
        except Exception:
            pass
    # Fallback
    return _heuristic_schema(instruction_text)
