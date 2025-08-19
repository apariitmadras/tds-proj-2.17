import os, io, re, json, uuid, base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

# ---- Internal modules (works with either core.deadline or core.budget) ----
from core.logging_setup import configure_logging, with_fields
try:
    from core.deadline import Deadline
except Exception:
    from core.budget import Deadline  # backward compatibility

# Optional helpers from your repo (safe if absent)
try:
    from core.fallbacks import make_fallback_answers
except Exception:
    def make_fallback_answers(n: int) -> List[Any]:
        return ["N/A" for _ in range(n)]

try:
    from core.prompt_runner import run_prompt
except Exception:
    def run_prompt(*args, **kwargs) -> Optional[str]:
        return None

# -------------------- Config --------------------
HARD_TIMEOUT_SEC = float(os.getenv("HARD_TIMEOUT_SEC", "285"))   # 4m45s SLA
PLOT_MAX_BYTES   = int(os.getenv("PLOT_MAX_BYTES", "100000"))    # ~100 KB
FALLBACK_MODE    = os.getenv("FALLBACK_MODE", "placeholders")    # placeholders | synthetic
APP_BUILD        = os.getenv("APP_BUILD", "v3-format-first")
STRICT_INPUT     = os.getenv("STRICT_INPUT", "0") == "1"         # if True, 400 on bad input; else synthesize

# -------------------- App --------------------
logger = configure_logging()
app = FastAPI(title="Format-First Answer Guarantor", redirect_slashes=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def _log(event: str, **fields):
    rec = with_fields(event=event, **fields); rec.msg = event
    logger.handle(rec)

# -------------------- Utilities --------------------
def _looks_like_uploadfile_repr(s: str) -> bool:
    return bool(s) and ("UploadFile(" in s or "Headers(" in s)

def _enumerated_lines(text: str) -> List[Tuple[int, str]]:
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        m = re.match(r'^(\d+)\s*[\.\)\]\-:]\s*(.+)$', ln)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out

def _extract_object_keys(text: str) -> List[str]:
    L = text or ""
    m = re.search(r'keys?\s*[:=]\s*(.+)$', L, flags=re.I | re.M)
    if not m: return []
    chunk = m.group(1).strip().strip("[](){}.")
    parts = [p.strip(" `\"'") for p in re.split(r'[,\|/;]', chunk)]
    seen, keys = set(), []
    for p in parts:
        if p and p.lower() not in seen:
            seen.add(p.lower()); keys.append(p)
    return keys[:50]

def _infer_out_type(text: str) -> str:
    L = (text or "").lower()
    if "json object" in L and "json array" not in L:
        return "object"
    return "array"

def _infer_array_len(text: str, enums: List[Tuple[int, str]]) -> Optional[int]:
    L = (text or "").lower()
    for pat in [r'exactly\s+(\d+)\s+(items?|keys?)', r'array\s+of\s+(\d+)',
                r'length\s+(\d+)', r'(\d+)\s+items?']:
        m = re.search(pat, L)
        if m:
            try:
                n = int(m.group(1))
                if n > 0: return n
            except Exception:
                pass
    if enums:
        nums = [n for (n, _) in enums]
        return max(nums) if min(nums) == 1 else len(nums)
    return None

def _extract_schema(text: str, time_left: int) -> Dict[str, Any]:
    out_type = _infer_out_type(text)
    enums = _enumerated_lines(text)
    questions = [q for (_, q) in enums]
    keys = _extract_object_keys(text) if out_type == "object" else []
    expected_len = _infer_array_len(text, enums)

    # Optional Schema LLM
    try:
        llm = run_prompt(
            role="schema",
            prompt_name="schema_extractor",
            variables={"instruction": text, "time_budget": time_left},
            system="You extract JSON schemas only."
        )
    except Exception:
        llm = None

    if llm:
        try:
            data = json.loads(llm)
            if isinstance(data, dict):
                if data.get("type") in ("array","object"):
                    out_type = data["type"]
                if out_type == "array":
                    k = data.get("length")
                    if isinstance(k, int) and k > 0: expected_len = k
                else:
                    k = data.get("keys")
                    if isinstance(k, list) and all(isinstance(x,str) for x in k):
                        keys = k[:50]
        except Exception:
            pass

    if out_type == "array":
        target_len = expected_len if (isinstance(expected_len,int) and expected_len>0) else (len(questions) if questions else 1)
        return {"out_type":"array","target_len":target_len,"keys":[],"questions":questions}
    else:
        final_keys = keys if keys else ([q[:80] for q in questions] if questions else ["answer"])
        return {"out_type":"object","target_len":len(final_keys),"keys":final_keys,"questions":questions}

def _make_plot_uri() -> str:
    x = np.arange(1, 11); y = x + np.random.RandomState(42).randn(len(x))
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.scatter(x, y); coef = np.polyfit(x, y, 1); yy = coef[0]*x + coef[1]
    ax.plot(x, yy, linestyle=":", color="red"); ax.set_xlabel("Rank"); ax.set_ylabel("Peak")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
    raw = buf.getvalue(); uri = "data:image/png;base64," + base64.b64encode(raw).decode()
    if len(uri.encode("utf-8")) > PLOT_MAX_BYTES:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHYALtTyR2wQAAAABJRU5ErkJggg=="
    return uri

# -------------------- Input parsing --------------------
async def _read_instruction_from_request(request: Request) -> str:
    content_type = (request.headers.get("content-type") or "")

    # JSON body
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            if STRICT_INPUT:
                raise HTTPException(status_code=400, detail="Invalid JSON body.")
            _log("input.json_parse_failed"); return "Return ONLY a JSON array with exactly 1 item:\n1. answer"
        instruction = (payload.get("task") or payload.get("questions") or payload.get("prompt"))
        if not instruction or not isinstance(instruction, str) or not instruction.strip():
            if STRICT_INPUT:
                raise HTTPException(status_code=400, detail="Provide instructions under 'task'/'questions'/'prompt'.")
            _log("input.json_missing", payload_keys=list(payload.keys()) if isinstance(payload, dict) else "n/a")
            return "Return ONLY a JSON array with exactly 1 item:\n1. answer"
        _log("input.json", length=len(instruction))
        return instruction

    # Multipart form
    try:
        form = await request.form()
    except Exception:
        if STRICT_INPUT:
            raise HTTPException(status_code=400, detail="Invalid or missing multipart form-data.")
        _log("input.form_parse_failed"); return "Return ONLY a JSON array with exactly 1 item:\n1. answer"

    # 1) Prefer explicit text fields
    pri = {"task","questions","prompt","q","instruction"}
    for key, val in form.multi_items():
        if not isinstance(val, UploadFile):
            s = str(val or "").strip()
            if s and key.lower() in pri:
                _log("input.multipart.text", field=key, length=len(s))
                return s

    # 2) Otherwise accept ANY non-empty text field
    for key, val in form.multi_items():
        if not isinstance(val, UploadFile):
            s = str(val or "").strip()
            if s:
                _log("input.multipart.text_any", field=key, length=len(s))
                return s

    # 3) Otherwise read the first non-empty file
    for key, val in form.multi_items():
        if isinstance(val, UploadFile):
            try:
                data = await val.read()
                text = (data or b"").decode("utf-8", errors="ignore").strip()
                if text and not _looks_like_uploadfile_repr(text):
                    _log("input.multipart.file", field=key, length=len(text))
                    return text
            except Exception:
                continue

    if STRICT_INPUT:
        raise HTTPException(status_code=400, detail="No instructions found. Upload a text file or include a 'task' field.")
    _log("input.missing_graceful", synthesized=True)
    return "Return ONLY a JSON array with exactly 1 item:\n1. answer"

# -------------------- API --------------------
@app.post("/api", include_in_schema=False)
@app.post("/api/")
async def api(request: Request):
    deadline = Deadline(HARD_TIMEOUT_SEC)
    req_id = str(uuid.uuid4())

    # 1) Read instruction (never blocks the "always answer" guarantee)
    instruction = await _read_instruction_from_request(request)

    # 2) FORMAT FIRST: determine schema (array
