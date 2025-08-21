# Data Analyst Agent

A  Data Analyst Agent that strictly obeys the output format requested in `questions.txt` and **discloses** synthetic mode via `X-Synthetic: true`.  
Designed to test format compliance, timeouts, and base64 image constraints—**without** real data processing.

## Features
- **Strict format compliance** (arrays with exact N items; objects with exact keys and order).
- **Synthetic image** generator (base64 PNG data URI; dotted red line; compact).
- **Multi-LLM capable** planner hooks (optional, disabled by default).
- **Railway-ready** (`/api/`, `questions.txt`, `/healthz`).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PORT=8080
python app.py
```

Healthcheck:
```bash
curl -s http://localhost:8080/healthz
```

## API
**POST** `/api/`  
Content-Type: `multipart/form-data` with file field **exactly** `questions.txt`

Use **mock mode**:
- query param `?mode=mock` or
- header `X-Mock: true`

Server sets `X-Synthetic: true` in the response.

### Examples

**Array with 1 item**
```bash
printf 'Return only a JSON array of strings with exactly 1 item:\n1. ok\n' > questions.txt
curl -s -F "questions.txt=@questions.txt" "http://localhost:8080/api/?mode=mock"
# => ["Synthetic value 1"]
```

**Object with keys & image limit**
```bash
cat > questions.txt << 'EOF'
Return a JSON object with keys: slope, min_year, max_year, image
The "image" must be a base-64 PNG data URI under 100,000 bytes.
EOF
curl -s -F "questions.txt=@questions.txt" "http://localhost:8080/api/?mode=mock" | jq .
```

## Environment variables
- `PORT` (Railway injects)
- `TIMEOUT_SECONDS` (default 285)
- `LOG_LEVEL` (`INFO` default)

Optional (for future multi-LLM planner integration):
- `OPENAI_API_KEY_PLANNER`
- `PLANNER_MODEL`

## Deployment (Railway)
- Push to GitHub; create a Railway service from repo.
- The Dockerfile and `railway.toml` are provided.
- Healthcheck path: `/healthz`.
