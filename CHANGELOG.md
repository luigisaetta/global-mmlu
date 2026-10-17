# Changelog

## Unreleased

* 2026-10-17: The UI also works when opened at `http://127.0.0.1:3000`
  (Next.js `allowedDevOrigins` and backend CORS defaults); the running
  accuracy line stays visible at 0% and 100%.
* 2026-10-17: `backend/start.sh` and `frontend/start.sh` start the two servers
  from a terminal, with prerequisite checks.
* 2026-10-17: Spec 001. FastAPI backend that samples Global-MMLU questions per
  language, evaluates an OCI Generative AI model, and streams progress over
  SSE; Next.js UI to pick model, language, and number of questions and watch
  progress and running accuracy.
