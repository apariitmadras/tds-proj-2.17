# llm.py (drop-in)
import os, asyncio
from typing import Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Allow separate keys per role; fall back to OPENAI_API_KEY
MODEL_KEYS = {
    "schema": os.getenv("OPENAI_API_KEY_SCHEMA") or os.getenv("OPENAI_API_KEY"),
    "guarantor": os.getenv("OPENAI_API_KEY_GUARANTOR") or os.getenv("OPENAI_API_KEY"),
    "synth": os.getenv("OPENAI_API_KEY_SYNTH") or os.getenv("OPENAI_API_KEY"),
}

# Allow per-role models; tweak in Railway if desired
MODELS = {
    "schema": os.getenv("OPENAI_MODEL_SCHEMA", "gpt-4o-mini"),
    "guarantor": os.getenv("OPENAI_MODEL_GUARANTOR", "gpt-4o-mini"),
    "synth": os.getenv("OPENAI_MODEL_SYNTH", "gpt-4o-mini"),
}

def _client(role: str) -> Optional[OpenAI]:
    if OpenAI is None:
        return None
    key = MODEL_KEYS.get(role)
    if not key:
        return None
    # Rely on environment proxies (HTTP(S)_PROXY) if present; no kwargs
    return OpenAI(api_key=key)

def _chat_sync(cli: OpenAI, model: str, system: str, user: str) -> str:
    r = cli.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content

async def ask(role: str, system: str, user: str, timeout: float = 10.0) -> Optional[str]:
    """
    Ask an OpenAI model for content. Returns None on any error or timeout.
    """
    cli = _client(role)
    if cli is None:
        return None
    model = MODELS.get(role) or "gpt-4o-mini"
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_chat_sync, cli, model, system, user),
            timeout=timeout,
        )
    except Exception:
        return None
