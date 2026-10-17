# Spec 001: Multilingual LLM evaluation with Global-MMLU

Status: implemented; locally tested and live-verified on OCI (see Verification).

## Problem

We want to measure how well an LLM hosted on OCI Generative AI answers
multiple-choice questions in different languages, using the Global-MMLU
dataset, from a local UI that shows progress and accuracy while the run
is in progress.

## Scope

* A local Python backend (FastAPI) that:
  * lists the supported languages and the configured models;
  * samples N random questions of one language from the `test` split;
  * calls the selected model once per question and grades the answer;
  * streams progress and running accuracy to the client (Server-Sent Events);
  * saves each completed run as a JSON file.
* A local Next.js UI to select model, language, and number of questions,
  start and stop a run, and watch progress, accuracy, and per-question results.

## Non-goals (for now)

* Dynamic model discovery (the model list, including dedicated endpoint
  OCIDs, is a configuration value).
* Authentication methods other than the OCI user principal (API key in
  `~/.oci/config`).
* Non-GENERIC chat formats (for example `CohereChatRequest`).
* Few-shot prompting with the `dev` split, chain-of-thought evaluation,
  subject filters, and cross-run comparison views.
* Deployment outside the developer machine.

## Assumptions and prerequisites

* macOS or Linux with Python 3.12 (Conda env `global-mmlu`) and Node.js 20+.
* An OCI config file with a profile whose user can call Generative AI
  inference in `eu-frankfurt-1` (policy `use generative-ai-family`).
* Outbound HTTPS to `huggingface.co` (dataset) and to the OCI inference endpoint.

## Dataset access

Verified 2026-10-17 against the Hugging Face Hub and dataset viewer APIs:

* Dataset `CohereLabs/Global-MMLU`, license Apache-2.0, not gated.
* 42 configs (one per language): am ar bn cs de el en es fa fil fr ha he hi
  id ig it ja ko ky lt mg ms ne nl ny pl pt ro ru si sn so sr sv sw te tr uk
  vi yo zh.
* Each config has a `test` split of 14,327 rows (about 4–6 MB of Parquet) and
  a `dev` split.
* Columns used: `sample_id`, `subject`, `subject_category`, `question`,
  `option_a`..`option_d`, `answer` (A–D), `cultural_sensitivity_label`
  (`CS`, `CA`, or `-` when not annotated).
* Parquet file URLs per config and split come from
  `GET https://huggingface.co/api/datasets/CohereLabs/Global-MMLU/parquet/{config}`.

Decision: download only the Parquet file of the selected language on first
use and cache it under `DATA_CACHE_DIR`. Rationale: a few MB per language,
exact uniform sampling, reproducible samples with a seed, and offline reruns.
Row-level streaming via the dataset viewer `/rows` API was rejected because
it needs one request per question and depends on rate limits.

## Model invocation

Verified 2026-10-17 with OCI Python SDK 2.188.0:

* `GenerativeAiClient.list_models` in `eu-frankfurt-1` lists
  `openai.gpt-5.6-luna` (capability `CHAT`, `ACTIVE`).
* `GenerativeAiInferenceClient.chat` with `OnDemandServingMode(model_id=...)`
  and `GenericChatRequest(api_format="GENERIC", max_completion_tokens=...,
  reasoning_effort="LOW")` returns the expected answer for this model.
* Dedicated AI cluster endpoints are called with
  `DedicatedServingMode(endpoint_id=<endpoint OCID>)` and the same
  `GenericChatRequest`; the backend chooses it when the target starts with
  `ocid1.generativeaiendpoint.`.
* Endpoint: `https://inference.generativeai.{region}.oci.oraclecloud.com`.
* Compartment: `OCI_COMPARTMENT_ID`, defaulting to the profile tenancy.

Reference: <https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm>

Prompt (fixed, English instruction, question and options in the target
language):

```
The following is a multiple choice question. Reply with only the letter
(A, B, C, or D) of the correct answer.

<question>

A. <option_a>
B. <option_b>
C. <option_c>
D. <option_d>
```

Grading: the first standalone letter A–D in the reply is the prediction.
A reply without one is recorded as `invalid` and counted as wrong.

## Intended behavior

API (all JSON, base `http://localhost:8000`):

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/models` | configured model IDs and the default |
| GET | `/api/languages` | language codes and English names |
| POST | `/api/evaluations` | SSE stream of a run |

`POST /api/evaluations` body: `model_id`, `language`, `num_questions`
(1–`MAX_QUESTIONS`), optional `seed`. Unknown model or language returns 422
before any remote call.

SSE events (`data:` lines carrying JSON with a `type` field):

* `status`: preparation messages (for example dataset download).
* `start`: `run_id`, `seed`, `total`, `model_id`, `language`.
* `result`: one per question, in completion order: `index`, `sample_id`,
  `subject`, `subject_category`, `cultural_sensitivity_label`, `expected`,
  `predicted`, `correct`, `invalid`, `latency_s`, `error`, plus running
  `completed`, `correct_count`, and `accuracy`.
* `done`: final summary with overall accuracy, accuracy by
  `subject_category` and by cultural sensitivity label, invalid and error
  counts, mean latency, `stopped: false`, and the path of the saved result file.
* `error`: fatal error; the stream ends.

Calls run concurrently, bounded by `MAX_CONCURRENCY`. A failed call (after the
SDK retry strategy) is recorded with `error` and counted as wrong; the run
continues. If the client disconnects (Stop button), no new calls are started
and the partial results are saved with `stopped: true`.

## Configuration

Environment variables (`.env`, see `.env.example`): `OCI_CONFIG_FILE`,
`OCI_PROFILE`, `OCI_REGION`, `OCI_COMPARTMENT_ID`, `MODELS`, `DEFAULT_MODEL`,
`MAX_QUESTIONS`, `MAX_CONCURRENCY`, `MAX_COMPLETION_TOKENS`, `DATA_CACHE_DIR`,
`RESULTS_DIR`, `CORS_ORIGINS`. Frontend: `NEXT_PUBLIC_API_BASE_URL`.

`MODELS` entries are `model_id` or `label=target`; the target is a model ID or
a dedicated endpoint OCID. The API and UI use the label; results record both
`model_id` (label) and `model_target`. Duplicate or empty labels fail at
startup.

## Costs and cleanup

Each question is one inference call: billed per token on demand, or covered
by the hourly cost of an existing dedicated AI cluster. No OCI resources are
created or changed. Local artifacts: `data/cache/` and `results/`
(both ignored by Git) can be deleted at any time.

## Acceptance criteria

1. The UI lists the configured model, the 42 languages, and accepts N.
2. Starting a run shows a progress bar and running accuracy that update as
   answers arrive, and a table of per-question results.
3. The same seed, language, and N select the same questions.
4. Stop ends the stream, no further calls start, and partial results are saved.
5. Invalid inputs are rejected with 422 without remote calls.
6. Unit tests run without network or credentials.

## Verification

Local checks, 2026-10-17 (macOS, Python 3.12 in Conda env `global-mmlu`,
OCI SDK 2.188.0, FastAPI, Node.js 24, Next.js 16.4.2):

* `pytest`: 24 passed (fake model and local Parquet fixtures, no network).
* Black: clean. Pylint: 10.00/10. ESLint: clean. `next build`: succeeded.

Live checks on OCI Generative AI, 2026-10-17, `eu-frankfurt-1`, user principal
(profile with an API key), model `openai.gpt-5.6-luna`, `MAX_CONCURRENCY=4`:

| Check | Observed |
| --- | --- |
| API, `it`, N=5, seed 42 | `status` (download), `start`, 5 `result`, `done`; 4/5 correct (0.80) |
| UI, `de`, N=20 | live progress, tiles, chart; 15/20 (0.75); breakdown shown; file saved |
| UI Stop, `de`, N=100 | stopped at 34/100; Start re-enabled |
| API disconnect after 6 s, `it`, N=100, seed 7 | 18 results received; backend logged the stop and saved them with `stopped: true` |

| Dedicated endpoint (`md-gptoss120b-fusion-endpoint`, compartment `dac`, imported gpt-oss-120b model), API, `it`, N=10, seed 42 | 10/10 correct, 0 invalid, 0 errors, mean latency 2.7 s; calls used `DedicatedServingMode` |

Before the dedicated run, single probe calls on that endpoint succeeded with
both parameter sets (`max_tokens`+`temperature` and
`max_completion_tokens`+`reasoning_effort`) and with the compartment set to
either the tenancy or the endpoint's compartment.

Mean latency was about 1.0–1.1 s per call on demand. These are single runs on small
samples: they verify the mechanism, not the model's accuracy.

Not verified: other on-demand vendors (non-OpenAI models), Cohere formats, other regions, runs near
`MAX_QUESTIONS`, behavior under service throttling (429).
