import os, asyncio
from typing import Optional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Model-specific keys
MODEL_KEYS = {
    "schema": os.getenv("OPENAI_API_KEY_SCHEMA") or os.getenv("OPENAI_API_KEY"),
    "guarantor": os.getenv("OPENAI_API_KEY_GUARANTOR") or os.getenv("OPENAI_API_KEY"),
    "synth": os.getenv("OPENAI_API_KEY_SYNTH") or os.getenv("OPENAI_API_KEY"),
}

MODELS = {
    "schema": os.getenv("OPENAI_MODEL_SCHEMA", "gpt-4o-mini"),
    "guarantor": os.getenv("OPENAI_MODEL_GUARANTOR", "gpt-4o-mini"),
    "synth": os.getenv("OPENAI_MODEL_SYNTH", "gpt-4o-mini"),
}

def _client(role: str) -> Optional[OpenAI]:
    if OpenAI is None: return None
    key = MODEL_KEYS.get(role)
    if not key: return None
    return OpenAI(api_key=key)

async def ask(role: str, system: str, user: str, timeout: float = 10.0) -> Optional[str]:
    """
    Minimal async wrapper. If OpenAI is unavailable or times out, return None (caller must fallback).
    """
    cli = _client(role)
    if cli is None: return None

    async def _do():
        # Use Chat Completions for broad compatibility
        r = cli.chat.completions.create(
            model=MODELS[role],
            temperature=0.0,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
        )
        return r.choices[0].message.content

    try:
        return await asyncio.wait_for(asyncio.to_thread(lambda: asyncio.run(_do()) if False else _do()), timeout=timeout)
    except Exception:
        return None
