# A2A Invoice Action Agent (starter)

## What this includes
- FastAPI HTTP+JSON routes and public Agent Card.
- Bearer-token protection for task routes.
- SQLite persistence, per-principal task ownership, and message-ID idempotency.
- One batched OpenAI-compatible model call per new batch, if configured.
- Proposal/result separation: no payment integration exists; accepted actions are recorded as receipts only.
- Conservative `open_exception` fallback if no model key is configured.

## Important limitations before graded use
This is a deployable **starter**, not a verified full-score solution. The challenge's exact invoice-package schema and evidence paragraphs must be tested against the grader. It does not implement distributed locking for multiple server instances, a fully atomic cancel/result transaction, a durable model-decision cache keyed by canonical package content, or every A2A error-envelope detail. Use one instance with persistent disk for initial testing and harden those areas before spending scarce checks. Never connect this sample to a real payment rail.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export A2A_BEARER_TOKEN='the-exact-grader-token'
export A2A_BASE_URL='https://YOUR-HOST/a2a/'
uvicorn main:app --host 0.0.0.0 --port 8000
```

Set `OPENAI_API_KEY` to enable the model. Optionally set `OPENAI_BASE_URL` and `OPENAI_MODEL` for an OpenAI-compatible provider. Keep keys in hosting environment variables, never in source control.

## Routes
- `GET /.well-known/agent-card.json`
- `POST /a2a/message:send`
- `GET /a2a/tasks`
- `GET /a2a/tasks/{id}`
- `POST /a2a/tasks/{id}:cancel`
- `GET /healthz`

## Deploy to Render
1. Push this folder to a GitHub repository.
2. Create a Render Web Service from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Configure `A2A_BEARER_TOKEN`, `A2A_BASE_URL`, `GEMINI_API_KEY`, and optionally `GEMINI_MODEL` in Environment.
6. Attach persistent disk for `tasks.db`; otherwise a redeploy may lose task state.
7. Confirm the Agent Card is reachable, then submit the base URL ending in `/a2a/`.

The base URL must exactly match the URL in `supportedInterfaces`, use HTTPS, and not redirect.
