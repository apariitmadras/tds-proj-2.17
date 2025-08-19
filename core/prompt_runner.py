import json, pathlib
from typing import Optional, Dict, Any, List
from .openai_client import call_openai

PROMPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "prompts"

def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")

def _render(template: str, variables: Dict[str, Any]) -> str:
    out = template
    for k, v in variables.items():
        placeholder = "{{ " + k + " }}"
        if isinstance(v, (dict, list)):
            out = out.replace(placeholder, json.dumps(v, ensure_ascii=False))
        else:
            out = out.replace(placeholder, str(v))
    return out

def run_prompt(role: str, prompt_name: str, variables: Dict[str, Any], system: Optional[str] = None) -> Optional[str]:
    tmpl = _load_prompt(prompt_name)
    user = _render(tmpl, variables)
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return call_openai(role, msgs)
