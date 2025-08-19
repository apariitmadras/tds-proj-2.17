import os, logging
from typing import Dict, Any, Optional
from jinja2 import Template

from .openai_client import call_openai

log = logging.getLogger(__name__)

PROMPTS_DIR = os.getenv("PROMPTS_DIR", os.path.join(os.path.dirname(__file__), "..", "prompts"))

def _load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{name}.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def run_prompt(role: str, prompt_name: str, variables: Dict[str, Any], system: Optional[str] = None) -> Optional[str]:
    raw = _load_prompt(prompt_name)
    txt = Template(raw).render(**variables)
    out = call_openai(role=role, prompt=txt, system=system)
    return out
