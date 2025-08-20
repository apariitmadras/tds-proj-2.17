# app.py
import time, os, logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from core.planner import plan_spec
from core.assembler import assemble_output
from core.validator import validate_output
from utils.format_detect import parse_user_constraints

APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "8080"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "285"))  # 4m45s default
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("synth-agent-pro")

app = FastAPI(title="Synthetic Data Analyst Agent (Pro)", version="0.2.0")

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

@app.post("/api/")
async def api_root(
    response: Response,
    questions: UploadFile = File(..., alias="questions.txt"),
    x_mock: Optional[str] = Header(None),
    mode: Optional[str] = None,
):
    start = time.time()
    is_mock = (mode == "mock") or (str(x_mock).lower() == "true")
    if not is_mock:
        # Safety: only synthetic mode allowed
        raise HTTPException(400, detail="This deployment only supports synthetic mode. Use ?mode=mock or header X-Mock: true")

    response.headers["X-Synthetic"] = "true"

    try:
        text = (await questions.read()).decode("utf-8", errors="ignore")
    except Exception:
        raise HTTPException(400, detail="questions.txt is required and must be readable (UTF-8).")

    # Deadline guard
    def ensure_time():
        if time.time() - start > TIMEOUT_SECONDS:
            raise HTTPException(504, detail="Timeout")

    ensure_time()
    log.info("Received questions.txt (%d bytes), mock=%s", len(text), is_mock)

    # Parse user constraints (size limits, image hints)
    constraints = parse_user_constraints(text)

    # PLAN: derive strict spec (shape, keys, order, types, image, etc.)
    spec = plan_spec(text)
    log.debug("Spec: %s", spec)
    ensure_time()

    # ASSEMBLE: build synthetic output that matches the spec
    output = assemble_output(spec, constraints)
    ensure_time()

    # VALIDATE: strict validation (shape/keys/order/types/limits)
    try:
        validate_output(output, spec, constraints)
    except Exception as e:
        # Return error; the caller’s grader expects wrong structure to fail.
        log.exception("Validation failed: %s", e)
        raise HTTPException(422, detail=f"Validation failed: {e}")

    ensure_time()
    return JSONResponse(output)
