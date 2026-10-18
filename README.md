# Global-MMLU multilingual evaluation

Local tool to measure how well an LLM on OCI Generative AI answers
multiple-choice questions in many languages, using
[Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU)
(42 languages, 14,327 test questions each) or
[Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite)
(23 languages, 400 test questions each, balanced between culturally sensitive
and culturally agnostic questions). Lite also covers languages missing from
the full set: Albanian, Burmese, Hungarian, Odia, Slovak, Tajik, and Welsh.

From a local web UI you pick the model, the dataset, the language, and the
number of questions. The backend samples that many random questions from the
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
| `MODELS`, `DEFAULT_MODEL` | models offered by the UI (see below) | `openai.gpt-5.6-luna` |
| `MAX_CONCURRENCY` | parallel calls per run | `4` |
| `DEFAULT_SEED` | seed pre-filled in the UI | `42` |
| `MAX_QUESTIONS` | upper bound per run | `500` |

`MODELS` is a comma-separated list. An entry is either an on-demand model ID,
or `label=target` to show a readable label in the UI. A target that is the
OCID of a dedicated AI cluster endpoint (`ocid1.generativeaiendpoint...`) is
called on that endpoint:

```
MODELS=openai.gpt-5.6-luna,my-gpt-oss=ocid1.generativeaiendpoint.oc1.eu-frankfurt-1.<id>
DEFAULT_MODEL=openai.gpt-5.6-luna
```

Restart the backend after changing `.env`. On-demand OpenAI models are sent
`max_completion_tokens`; other models and all dedicated endpoints get
`max_tokens` and `temperature=0`. Cohere models (`CohereChatRequest` format)
are not supported yet.

### Reasoning effort

The UI lets you choose the reasoning effort of a run: **Model default** (the
parameter is not sent) or `none`, `minimal`, `low`, `medium`, `high`. The
choice is saved with the results; use the same one when comparing models.

* Reasoning models can spend the whole `MAX_COMPLETION_TOKENS` budget on
  reasoning and return an empty answer, recorded as `invalid` with a latency
  close to the time needed to generate the budget. Lowering the effort (for
  example `none`) avoids it.
* Some models reject the parameter, for example `openai.gpt-4.1` for any value
  and some gpt-oss deployments for `none`. The run then stops at once with
  the service error; choose **Model default**. Others, such as Llama, ignore it.


## Run

In two terminals, from the repository root:

```bash
backend/start.sh
```

```bash
frontend/start.sh
```

Stop each server with `Ctrl+C`. `backend/start.sh --reload` restarts the
backend when Python code changes (edits to `.env` still need a manual
restart). Optional variables: `CONDA_ENV`, `HOST`, and `PORT` for the backend,
`PORT` for the UI (add a new UI origin to `CORS_ORIGINS`).

Open <http://localhost:3000>, choose model, language, and number of
questions, and press **Start evaluation**. The seed is required and is
pre-filled with `DEFAULT_SEED` (42): the same seed, dataset, language, and
number of questions always select the same questions, so keep it fixed when
comparing models. **Stop** ends the run; results received so far are kept.

Accuracy is shown with its 95% Wilson confidence interval: in the
Accuracy tile, as a shaded band in the running chart, and for each
category in the breakdown (bar whiskers and text).

The **Runs** page (<http://localhost:3000/runs>) lists every result file in
`results/`, newest first, with accuracy, interval, invalid and error counts,
latency, and status. Filter by model, dataset, language, and reasoning effort;
click a run to see its breakdowns and answers. Stopped (partial) runs are
hidden unless **Show stopped runs** is checked; their files are kept. Runs saved before the effort
was recorded show "not recorded" (on-demand OpenAI models then got `low`).

The first run of a language downloads its Parquet file (a few MB, less for
Lite) into `data/cache/<dataset>/`. Each run is saved as JSON in `results/`.
With Lite, 400 questions give a margin of error of about ±4–5 points (95%).

## API

```bash
curl -N -X POST http://localhost:8000/api/evaluations \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "openai.gpt-5.6-luna", "dataset": "global-mmlu-lite", "language": "sq", "num_questions": 5, "seed": 42}'
```

`dataset` is `global-mmlu` (default) or `global-mmlu-lite`.

The response is an SSE stream of `status`, `start`, `result`, and `done`
events (see the specification). Other endpoints: `GET /api/models`,
`GET /api/datasets`, `GET /api/languages?dataset=...`, `GET /api/runs`,
`GET /api/runs/{run_id}`, `GET /api/health`.

## Checks

```bash
conda run -n global-mmlu black --check backend tests
conda run -n global-mmlu pylint backend/app tests
conda run -n global-mmlu pytest
npm --prefix frontend run lint
```

Unit tests use local fixtures and a fake model: no network or credentials.

## Costs and cleanup

Each question is one inference call: billed per token on demand, or covered
by the hourly cost of an existing dedicated AI cluster. No OCI resources are
created. `data/` and `results/` are local and can be deleted.

## Troubleshooting

* **Cannot reach the backend** in the UI: start the backend, and check that
  `CORS_ORIGINS` includes the UI origin.
* **`OCI client setup failed`**: check `OCI_CONFIG_FILE` and `OCI_PROFILE`.
* **Every answer is `error`**: hover the cell to see the service error; a
  `404` usually means the model is not available in the region or compartment,
  `401`/`403` an authentication or policy problem.
