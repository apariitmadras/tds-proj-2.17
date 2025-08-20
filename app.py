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

# ---- shared handler so /api and /api/ behave identically ----
async def _handle_api_request(
    response: Response,
    questions: UploadFile,
    x_mock: Optional[str],
    mode: Optional[str],
):
    start = time.time()

    def ensure_time():
        if time.time() - start > TIMEOUT_SECONDS:
            raise HTTPException(504, detail="Timeout")

    # synthetic-only mode (explicitly disclosed)
    is_mock = (mode == "mock") or (str(x_mock).lower() == "true")
    if not is_mock:
        raise HTTPException(
            400,
            detail="This deployment only supports synthetic mode. Use ?mode=mock or header X-Mock: true",
        )
    response.headers["X-Synthetic"] = "true"

    # read questions.txt
    try:
        text = (await questions.read()).decode("utf-8", errors="ignore")
    except Exception:
        raise HTTPException(400, detail="questions.txt is required and must be readable (UTF-8).")

    ensure_time()
    log.info("Received questions.txt (%d bytes), mock=%s", len(text), is_mock)

    # derive constraints/spec
    constraints = parse_user_constraints(text)
    spec = plan_spec(text)
    log.debug("Spec: %s", spec)

    ensure_time()
    # assemble synthetic output
    output = assemble_output(spec, constraints)

    ensure_time()
    # validate structure strictly
    try:
        validate_output(output, spec, constraints)
    except Exception as e:
        log.exception("Validation failed: %s", e)
        raise HTTPException(422, detail=f"Validation failed: {e}")

    ensure_time()
    return JSONResponse(output)

# ---- accept BOTH /api/ and /api ----

@app.post("/api/")
async def api_root_slash(
    response: Response,
    questions: UploadFile = File(..., alias="questions.txt"),
    x_mock: Optional[str] = Header(None),
    mode: Optional[str] = None,
):
    return await _handle_api_request(response, questions, x_mock, mode)

@app.post("/api")
async def api_root_no_slash(
    response: Response,
    questions: UploadFile = File(..., alias="questions.txt"),
    x_mock: Optional[str] = Header(None),
    mode: Optional[str] = None,
):
    return await _handle_api_request(response, questions, x_mock, mode)
