import os, logging
from typing import Optional
from openai import OpenAI

log = logging.getLogger(__name__)
DEFAULT_MODEL = "gpt-4o-mini"

def _client_for(key_env: str) -> Optional[OpenAI]:
    api_key = os.getenv(key_env)
    if not api_key:
        log.warning("OpenAI key not set for %s", key_env)
        return None
    return OpenAI(api_key=api_key)

def _model_for(model_env: str) -> str:
    return os.getenv(model_env, DEFAULT_MODEL)

def call_openai(role: str, prompt: str, system: Optional[str] = None,
                temperature: float = 0.2, max_tokens: int = 1200) -> Optional[str]:
    role = role.lower()
    if role == "planner":
        client = _client_for("OPENAI_API_KEY_PLANNER");   model = _model_for("OPENAI_MODEL_PLANNER")
    elif role == "orch":
        client = _client_for("OPENAI_API_KEY_ORCH");      model = _model_for("OPENAI_MODEL_ORCH")
    elif role == "analyst":
        client = _client_for("OPENAI_API_KEY_ANALYST");   model = _model_for("OPENAI_MODEL_ANALYST")
    elif role == "validator":
        client = _client_for("OPENAI_API_KEY_VALIDATOR"); model = _model_for("OPENAI_MODEL_VALIDATOR")
    elif role == "guarantor":
        client = _client_for("OPENAI_API_KEY_GUARANTOR"); model = _model_for("OPENAI_MODEL_GUARANTOR")
    else:
        client = _client_for("OPENAI_API_KEY_ANALYST");   model = _model_for("OPENAI_MODEL_ANALYST")

    if client is None:
        return None

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.exception("OpenAI call failed for role=%s: %s", role, e)
        return None
