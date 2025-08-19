# Format-First Answer Guarantor (V3)

Primary objective: **Always** return a valid answer in the exact format the user requested (real if possible, synthetic if not) within **4m45s**.

Secondary objective: produce real answers using multiple OpenAI models (separate API keys per model).

## Deploy (Railway)

- **Start command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Env vars** (set in Railway → Variables):
  - `HARD_TIMEOUT_SEC=285`
  - `FALLBACK_MODE=synthetic`  # or `placeholders`
  - `APP_BUILD=v3-format-first`
  - `OPENAI_API_KEY_SCHEMA=...` `OPENAI_MODEL_SCHEMA=gpt-4o-mini`
  - `OPENAI_API_KEY_GUARANTOR=...` `OPENAI_MODEL_GUARANTOR=gpt-4o-mini`
  - (Optional realism) `OPENAI_API_KEY_PLANNER=...`, `..._ORCH=...`, `..._ANALYST=...`, `..._VALIDATOR=...`

## Endpoint

- **POST** `https://<your-railway>.app/api` (only POST)
- **GET** `https://<your-railway>.app/healthz` → `{"status":"ok","build":"..."}`

## Example request (curl)

```bash
printf "Return ONLY a JSON array with exactly 4 items:\n1. a\n2. b\n3. c\n4. d\n" > q.txt
curl -s -F "file=@q.txt;type=text/plain" https://<your-railway>.app/api
