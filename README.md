# Global-MMLU multilingual evaluation

Local tool to measure how well an LLM on OCI Generative AI answers
[Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU)
questions in each of its 42 languages.

From a local web UI you pick the model, the language, and the number of
questions. The backend samples that many random questions from the
language's `test` split, asks the model, and streams progress and running
accuracy back to the UI.

* `backend/`: FastAPI service (Python 3.12) that downloads and caches the
  dataset per language, calls OCI Generative AI, and streams Server-Sent Events.
* `frontend/`: Next.js UI.
* `specs/001-multilingual-evaluation.md`: scope, design decisions, API, and
  acceptance criteria.

## Prerequisites

* Conda and Node.js 20 or later.
* An OCI config file (default `~/.oci/config`) with a profile whose user can
  call Generative AI inference in `eu-frankfurt-1`, for example through the
  policy `allow group <group> to use generative-ai-family in compartment <compartment>`.
* Internet access to `huggingface.co` and to the OCI inference endpoint.

## Setup

```bash
conda create -n global-mmlu python=3.12
conda activate global-mmlu
pip install -r requirements-dev.txt
cp .env.example .env            # then set OCI_PROFILE (and OCI_COMPARTMENT_ID if needed)
cd frontend && npm install && cp .env.example .env.local
```

Main settings in `.env`:

| Variable | Meaning | Default |
| --- | --- | --- |
| `OCI_PROFILE` | profile of the OCI config file (user principal) | `DEFAULT` |
| `OCI_REGION` | region of the inference endpoint | `eu-frankfurt-1` |
| `OCI_COMPARTMENT_ID` | compartment for the calls | profile tenancy |
| `MODELS`, `DEFAULT_MODEL` | model IDs offered by the UI | `openai.gpt-5.6-luna` |
| `MAX_CONCURRENCY` | parallel calls per run | `4` |
| `MAX_QUESTIONS` | upper bound per run | `500` |

## Run

In two terminals, from the repository root:

```bash
conda run --no-capture-output -n global-mmlu uvicorn app.main:app --app-dir backend --port 8000
```

```bash
npm --prefix frontend run dev
```

Open <http://localhost:3000>, choose model, language, and number of
questions, and press **Start evaluation**. Set a seed to reuse the same
sample across models. **Stop** ends the run; results received so far are kept.

The first run of a language downloads its Parquet file (about 5 MB) into
`data/cache/`. Each run is saved as JSON in `results/`.

## API

```bash
curl -N -X POST http://localhost:8000/api/evaluations \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "openai.gpt-5.6-luna", "language": "it", "num_questions": 5, "seed": 42}'
```

The response is an SSE stream of `status`, `start`, `result`, and `done`
events (see the specification). Other endpoints: `GET /api/models`,
`GET /api/languages`, `GET /api/health`.

## Checks

```bash
conda run -n global-mmlu black --check backend tests
conda run -n global-mmlu pylint backend/app tests
conda run -n global-mmlu pytest
npm --prefix frontend run lint
```

Unit tests use local fixtures and a fake model: no network or credentials.

## Costs and cleanup

Each question is one on-demand inference call, billed per token. No OCI
resources are created. `data/` and `results/` are local and can be deleted.

## Troubleshooting

* **Cannot reach the backend** in the UI: start the backend, and check that
  `CORS_ORIGINS` includes the UI origin.
* **`OCI client setup failed`**: check `OCI_CONFIG_FILE` and `OCI_PROFILE`.
* **Every answer is `error`**: hover the cell to see the service error; a
  `404` usually means the model is not available in the region or compartment,
  `401`/`403` an authentication or policy problem.
