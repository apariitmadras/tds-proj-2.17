import os, logging
from typing import Optional, List, Dict, Any
from openai import OpenAI

log = logging.getLogger(__name__)
DEFAULT_MODEL = "gpt-4o-mini"

ROLE_TO_ENV = {
    "schema":    ("OPENAI_API_KEY_SCHEMA",    "OPENAI_MODEL_SCHEMA"),
    "guarantor": ("OPENAI_API_KEY_GUARANTOR", "OPENAI_MODEL_GUARANTOR"),
    "validator": ("OPENAI_API_KEY_VALIDATOR", "OPENAI_MODEL_VALIDATOR"),
    "planner":   ("OPENAI_API_KEY_PLANNER",   "OPENAI_MODEL_PLANNER"),
    "orch":      ("OPENAI_API_KEY_ORCH",      "OPENAI_MODEL_ORCH"),
    "analyst":   ("OPENAI_API_KEY_ANALYST",   "OPENAI_MODEL_ANALYST"),
}

def _client_for(key_env: str) -> Optional[OpenAI]:
    key = os.getenv(key_env)
    if not key:
        log.warning("OpenAI key not set for %s", key_env)
        return None
    return OpenAI(api_key=key)

def _model_for(model_env: str) -> str:
    return os.getenv(model_env, DEFAULT_MODEL)

def call_openai(role: str, messages: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: int = 1200) -> Optional[str]:
    key_env, model_env = ROLE_TO_ENV.get(role, ("OPENAI_API_KEY_ANALYST", "OPENAI_MODEL_ANALYST"))
    client = _client_for(key_env)
    if client is None: return None
    try:
        resp = client.chat.completions.create(
            model=_model_for(model_env),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.exception("OpenAI call failed for role=%s: %s", role, e)
        return None
