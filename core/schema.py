import json
from typing import Dict, Any
from .format_enforcer import parse_request_structure
from .prompt_runner import run_prompt

def extract_schema(text: str, deadline_seconds: int) -> Dict[str, Any]:
    """
    Ask Schema LLM for the desired structure; merge with heuristics; return final schema:
      { out_type: "array"|"object", target_len: int, keys: [str], questions: [str] }
    Always returns a usable schema even if LLM fails.
    """
    heur = parse_request_structure(text)
    out_type = heur["out_type"]
    questions = heur["questions"]
    expected_len = heur["expected_len"]
    keys = heur["obj_keys"]

    # LLM pass (best-effort and time-aware)
    llm = run_prompt(
        role="schema",
        prompt_name="schema_extractor",
        variables={"instruction": text, "time_budget": deadline_seconds},
        system="You extract JSON schemas only."
    )

    if llm:
        try:
            data = json.loads(llm)
            if isinstance(data, dict):
                if data.get("type") in ("array", "object"):
                    out_type = data["type"]
                if out_type == "array":
                    k = data.get("length")
                    if isinstance(k, int) and k > 0: expected_len = k
                else:
                    k = data.get("keys")
                    if isinstance(k, list) and all(isinstance(x, str) for x in k):
                        keys = k[:50]
        except Exception:
            pass

    # Final shape
    if out_type == "array":
        target_len = expected_len if isinstance(expected_len, int) and expected_len > 0 else (len(questions) if questions else 1)
        return {"out_type": "array", "target_len": target_len, "keys": [], "questions": questions}
    else:
        final_keys = keys if keys else ([q[:80] for q in questions] if questions else ["answer"])
        return {"out_type": "object", "target_len": len(final_keys), "keys": final_keys, "questions": questions}
