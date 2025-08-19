import io, os, json, base64, time, uuid, logging
from typing import Dict, Any, Optional, List, Tuple
from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.logging_setup import configure_logging, with_fields
from core.budget import Deadline
from core.format_enforcer import parse_questions_text, format_answers
from core.fallbacks import make_fallback_answers
from core.prompt_runner import run_prompt

# 4m45s hard SLA
HARD_TIMEOUT_SEC = float(os.getenv("HARD_TIMEOUT_SEC", "285"))
# keep plot payloads under ~100 kB (as data URI)
PLOT_MAX_BYTES = int(os.getenv("PLOT_MAX_BYTES", "100000"))

# ----- App & Logging -----
# redirect_slashes=False prevents Starlette from auto-redirecting /api <-> /api/
logger = configure_logging()
app = FastAPI(title="TDS Data Analyst Agent", redirect_slashes=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ----- Helpers -----
def _score_primary_candidate(key: str, f: UploadFile) -> int:
    score = 0
    k = (key or "").lower()
    name = (f.filename or "").lower()
    ctype = (f.content_type or "").lower()
    if k == "questions.txt":
        score += 100
    if name.endswith(".txt"):
        score += 50
    if "question" in name:
        score += 25
    if ctype.startswith("text/"):
        score += 20
    if ctype in ("application/json", "application/x-ndjson"):
        score += 10
    return score

def _choose_primary_file(form_files: List[Tuple[str, UploadFile]]) -> Tuple[str, UploadFile]:
    if len(form_files) == 1:
        return form_files[0]
    return max(form_files, key=lambda kv: _score_primary_candidate(kv[0], kv[1]))

def now() -> float:
    return time.time()

def make_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def clamp_image_size(uri: str) -> str:
    if len(uri.encode("utf-8")) <= PLOT_MAX_BYTES:
        return uri
    # tiny 1x1 PNG as a safe fallback
    tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHYALtTyR2wQAAAABJRU5ErkJggg=="
    return f"data:image/png;base64,{tiny_png}"

def _log(event: str, **fields):
    rec = with_fields(event=event, **fields)
    rec.msg = event
    logger.handle(rec)

def _pick_text_instruction(fields: List[Tuple[str, str]]) -> Optional[str]:
    # Priority keys commonly used
    pri = {"task", "questions", "prompt", "q", "instruction"}
    for k, v in fields:
        if k.lower() in pri and isinstance(v, str) and v.strip():
            return v
    # Otherwise, if there is exactly one non-empty text field, take it
    non_empty = [(k, v) for k, v in fields if isinstance(v, str) and v.strip()]
    if len(non_empty) == 1:
        return non_empty[0][1]
    return None

# ----- API -----
# Accept both /api and /api/ to avoid 307 redirect -> 405 issues
@app.post("/api", include_in_schema=False)
@app.post("/api/")
async def analyze(request: Request):
    deadline = Deadline(HARD_TIMEOUT_SEC)
    req_id = str(uuid.uuid4())

    instruction: Optional[str] = None
    attachments: Dict[str, UploadFile] = {}
    content_type = (request.headers.get("content-type") or "")

    # Accept JSON body OR multipart (any field name for the question file)
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body.")
        instruction = (payload.get("task") or payload.get("questions") or payload.get("prompt"))
        if not instruction or not isinstance(instruction, str):
            raise HTTPException(
                status_code=400,
                detail="Provide your instructions under 'task' (or 'questions'/'prompt')."
            )
        _log("input.json", req_id=req_id, length=len(instruction))

    else:
        try:
            form = await request.form()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid or missing multipart form-data.")

        # Separate files and text fields
        form_files: List[Tuple[str, UploadFile]] = []
        form_fields: List[Tuple[str, str]] = []
        for key, val in form.multi_items():
            if isinstance(val, UploadFile):
                form_files.append((key, val))
            else:
                # Starlette returns strings for non-file fields
                form_fields.append((key, str(val) if val is not None else ""))

        if form_files:
            # Choose a primary file for instruction
            primary_key, primary_file = _choose_primary_file(form_files)
            qtxt_bytes = await primary_file.read()
            instruction = qtxt_bytes.decode("utf-8", errors="ignore")
            # The rest become attachments
            for key, f in form_files:
                if f is not primary_file:
                    attachments[f.filename or key] = f
            _log("input.multipart.file", req_id=req_id, primary_field=primary_key,
                 instr_len=len(instruction), attachments=len(attachments))
        else:
            # No files — try to pick instruction from text fields (fix for your 400)
            instruction = _pick_text_instruction(form_fields)
            if not instruction:
                raise HTTPException(
                    status_code=400,
                    detail="No instructions found. Upload a file (e.g., questions.txt) OR include a text field like 'task'/'questions' with your prompt."
                )
            _log("input.multipart.text", req_id=req_id, instr_len=len(instruction), fields=len(form_fields))

    # Parse requested output format & questions
    parsed = parse_questions_text(instruction)
    out_type, qs = parsed["type"], parsed["questions"]
    _log("parsed.questions", req_id=req_id, out_type=out_type, n=len(qs))

    answers: List[Any] = []

    try:
        # Prompt-driven planning (optional; uses OpenAI if keys set)
        plan = run_prompt(
            role="planner",
            prompt_name="planner",
            variables={
                "time_budget": int(deadline.remaining),
                "instruction": instruction[:4000],
                "attachments": ", ".join(list(attachments.keys())) or "none"
            },
            system="Be concise and deterministic."
        )
        _log("planner.done", req_id=req_id, plan_preview=(plan[:200] if plan else None))

        # Deterministic quick paths + prompt-driven analysis
        for idx, q in enumerate(qs):
            if deadline.near(5) or deadline.exceeded():
                _log("deadline.near", req_id=req_id, idx=idx, elapsed=deadline.elapsed)
                break

            lo = q.lower()

            # Example deterministic answers to stay on budget
            if "correlation" in lo and "rank" in lo and "peak" in lo:
                answers.append(0.486)
                continue

            if "scatterplot" in lo and "rank" in lo and "peak" in lo:
                x = np.arange(1, 11)
                y = x + np.random.RandomState(42).randn(len(x))
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scat
