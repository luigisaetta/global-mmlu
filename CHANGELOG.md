# Changelog

## Unreleased

* 2026-10-18: The Runs page hides stopped runs by default, with a
  "Show stopped runs" toggle.
* 2026-10-18: 95% Wilson confidence interval for accuracy (tile, running
  chart band, breakdown bars, saved results).
* 2026-10-18: Runs page with the history of saved results, filters by model,
  dataset, language, and reasoning effort, and run details.
* 2026-10-18: The seed is required for every run (no random seed); the UI
  pre-fills `DEFAULT_SEED` from `.env` (42).
* 2026-10-17: Dataset selection: Global-MMLU or Global-MMLU-Lite (adds
  Albanian and six other languages). Cached files move to
  `data/cache/<dataset>/`; files in the old location are downloaded again.
* 2026-10-17: Reasoning effort per run (UI selector and API field), not sent
  by default; `openai.*` models no longer get a forced `LOW`. A request
  rejected with HTTP 400 stops the run with the service error.
* 2026-10-17: Dedicated AI cluster endpoints: `MODELS` accepts `label=target`
  entries, and endpoint OCIDs are called with `DedicatedServingMode`.
* 2026-10-17: The UI also works when opened at `http://127.0.0.1:3000`
  (Next.js `allowedDevOrigins` and backend CORS defaults); the running
  accuracy line stays visible at 0% and 100%.
* 2026-10-17: `backend/start.sh` and `frontend/start.sh` start the two servers
  from a terminal, with prerequisite checks.
* 2026-10-17: Spec 001. FastAPI backend that samples Global-MMLU questions per
  language, evaluates an OCI Generative AI model, and streams progress over
  SSE; Next.js UI to pick model, language, and number of questions and watch
  progress and running accuracy.
