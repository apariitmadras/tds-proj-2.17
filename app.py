import os, io, re, json, uuid, base64, time, logging, sys, random
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException, Header, Response
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from format_detect import detect_schema  # LLM-first, heuristic fallback
from synth import synth_values, tiny_png_data_uri
from validator import enforce_shape_and_types

APP_BUILD        = os.getenv("APP_BUILD", "synth-v1")
HARD_TIMEOUT_SEC = float(os.getenv("HARD_TIMEOUT_SEC", "285"))   # 4m45s
STRICT_INPUT     = os.getenv("STRICT_INPUT", "0") == "1"
DIAG             = os.getenv("DIAG", "1") == "1"

# -------- Logging to stdout (Railway) --------
logger = logging.getLogger("app")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

def jlog(event: str, **fields):
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))

# -------- App / CORS --------
app = FastAPI(title="Synthetic Format-First API", redirect_slashes=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# -------- Robust UploadFile detection (duck-typing safe) --------
try:
    from fastapi import UploadFile as FastAPIUploadFile
except Exception:
    FastAPIUploadFile = None
try:
    from starlette.datastructures import UploadFile as StarletteUploadFile
except Exception:
    StarletteUploadFile = None

def _is_upload_file(val: Any) -> bool:
    if val is None: return False
    try:
        if FastAPIUploadFile and isinstance(val, FastAPIUploadFile): return True
    except Exception: pass
    try:
        if StarletteUploadFile and isinstance(val, StarletteUploadFile): return True
    except Exception: pass
    return all(hasattr(val, a) for a in ("filename", "read", "content_type"))

# -------- Instruction ingestion (JSON → preferred text → file → any text) --------
PREFERRED_TEXT_KEYS = {"task", "questions", "prompt", "q", "instruction", "task_text"}
PRIMARY_FILE_KEYS   = ("file", "questions.txt")

async def read_instruction(request: Request) -> str:
    ct = (request.headers.get("content-type") or "").lower()
    jlog("input.ct", content_type=ct)

    if "application/json" in ct:
        try:
            data = await request.json()
        except Exception:
            if STRICT_INPUT: raise HTTPException(400, "Invalid JSON body.")
            jlog("instr.synth", reason="json_parse_failed")
            return "Return ONLY a JSON array with exactly 1 item:\n1. answer"
        if isinstance(data, dict):
            for k in PREFERRED_TEXT_KEYS:
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    jlog("instr.json", field=k, length=len(v))
                    return v.strip()
        if STRICT_INPUT: raise HTTPException(400, "Provide instructions under: " + ", ".join(sorted(PREFERRED_TEXT_KEYS)))
        jlog("instr.synth", reason="json_missing_task")
        return "Return ONLY a JSON array with exactly 1 item:\n1. answer"

    # multipart/ form
    try:
        form = await request.form()
    except Exception:
        if STRICT_INPUT: raise HTTPException(400, "Invalid or missing multipart form-data.")
        jlog("instr.synth", reason="form_parse_failed")
        return "Return ONLY a JSON array with exactly 1 item:\n1. answer"

    # preferred text fields
    for k, v in form.multi_items():
        if not _is_upload_file(v):
            s = str(v or "").strip()
            if s and k.lower() in PREFERRED_TEXT_KEYS:
                jlog("instr.text_preferred", field=k, length=len(s)); return s

    # primary files
    for key in PRIMARY_FILE_KEYS:
        if key in form:
            f = form[key]
            if _is_upload_file(f):
                try:
                    data = await f.read()
                    text = (data or b"").decode("utf-8", errors="ignore").strip()
                    if text and "UploadFile(" not in text:
                        jlog("instr.file_primary", field=key, filename=getattr(f, "filename", None), length=len(text))
                        return text
                except Exception as e:
                    jlog("instr.file_primary_error", field=key, err=str(e))

    # any file
    for k, v in form.multi_items():
        if _is_upload_file(v):
            try:
                data = await v.read()
                text = (data or b"").decode("utf-8", errors="ignore").strip()
                if text and "UploadFile(" not in text:
                    jlog("instr.file_any", field=k, filename=getattr(v, "filename", None), length=len(text))
                    return text
            except Exception as e:
                jlog("instr.file_any_error", field=k, err=str(e))

    # any remaining non-trivial text
    for k, v in form.multi_items():
        if not _is_upload_file(v):
            s = str(v or "").strip()
            if s and k.lower() not in {"mode","mock","x-mock"} and len(s) >= 8:
                jlog("instr.text_any", field=k, length=len(s)); return s

    if STRICT_INPUT: raise HTTPException(400, "No instructions found.")
    jlog("instr.synth", reason="no_input")
    return "Return ONLY a JSON array with exactly 1 item:\n1. answer"

# -------- Deadline helper --------
class Deadline:
    def __init__(self, seconds: float):
        self.t0 = time.time(); self.budget = seconds
    @property
    def elapsed(self): return time.time() - self.t0
    @property
    def remaining(self): return max(0.0, self.budget - self.elapsed)
    def near(self, margin: float = 10.0): return self.remaining <= margin

# -------- Routes --------
@app.get("/healthz")
async def healthz():
    return JSONResponse({"status":"ok","build":APP_BUILD})

@app.post("/api", include_in_schema=False)
@app.post("/api/")
async def api(request: Request, response: Response, x_mock: Optional[str] = Header(None), mode: Optional[str] = None):
    dl = Deadline(HARD_TIMEOUT_SEC)
    req_id = str(uuid.uuid4())
    seed = req_id[:8]

    # 1) read instruction
    instruction = await read_instruction(request)

    # 2) extract schema (LLM-first → heuristic fallback)
    schema = await detect_schema(instruction, timeout=min(30, max(6, dl.remaining/2)))
    # schema = { "out_type": "array"|"object", "target_len": int, "keys": [...], "hints": [...], "questions": [...] }

    # 3) synthesize typed values (synthetic only; no "N/A")
    values = synth_values(schema, seed=seed)

    # 4) enforce exact format/shape/types
    final_payload = enforce_shape_and_types(schema, values)

    jlog("response.ok", build=APP_BUILD, elapsed_ms=int(dl.elapsed*1000), bytes=len(json.dumps(final_payload)))
    return JSONResponse(content=final_payload)

# Optional diagnostics (echo what server receives)
@app.post("/__echo")
async def __echo(request: Request):
    if not DIAG:
        return JSONResponse({"detail":"diag disabled; set DIAG=1"}, status_code=403)
    info = {"content_type": request.headers.get("content-type"), "fields": []}
    try:
        form = await request.form()
    except Exception as e:
        return JSONResponse({"error": f"form-parse-failed: {e}"})
    for k, v in form.multi_items():
        if _is_upload_file(v):
            data = await v.read()
            info["fields"].append({"name":k,"kind":"file","filename":getattr(v,"filename",None),"ctype":getattr(v,"content_type",None),"bytes":len(data),"text_preview":(data.decode('utf-8','ignore')[:120] if data else "")})
        else:
            s = str(v or "")
            info["fields"].append({"name":k,"kind":"text","len":len(s),"value_preview":s[:120]})
    return JSONResponse(info)
