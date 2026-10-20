# Spec 001: Multilingual LLM evaluation with Global-MMLU and Global-MMLU-Lite

Status: implemented; locally tested and live-verified on OCI (see Verification).

## Problem

We want to measure how well an LLM hosted on OCI Generative AI answers
multiple-choice questions in different languages, using the Global-MMLU
dataset, from a local UI that shows progress and accuracy while the run
is in progress.

## Scope

* A local Python backend (FastAPI) that:
  * lists the configured models and the supported datasets with their languages;
  * samples N random questions of one language from the `test` split of the
    selected dataset (Global-MMLU or Global-MMLU-Lite);
  * calls the selected model once per question and grades the answer;
  * streams progress and running accuracy to the client (Server-Sent Events);
  * saves each completed run as a JSON file.
* A local Next.js UI to select model, dataset, language, and number of questions,
  start and stop a run, and watch progress, accuracy with its 95% confidence
  interval, and per-question results.
* A Runs page that lists the saved result files with filters, sorted by date
  (newest first, default), by decreasing accuracy, or by decreasing lower
  bound of the 95% interval (ties: higher accuracy, then newer first), and
  shows the details of one run. Stopped runs are hidden by default (toggle to show
  them); hiding never deletes files.

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

Global-MMLU-Lite, verified 2026-10-17 with the same APIs:

* Dataset `CohereLabs/Global-MMLU-Lite`, license Apache-2.0, not gated, same
  columns as Global-MMLU.
* 23 configs: ar bn cs cy de en es fr hi hu id it ja ko my or pt sk sq sw tg
  yo zh. `cy`, `hu`, `my`, `or`, `sk`, `sq` (Albanian), and `tg` are not in
  Global-MMLU.
* Each config has a `test` split of 400 rows (for `sq`, checked: 200 `CS`,
  200 `CA`); `cy` has no `dev` split.

Global-MMLU has no Albanian config. Other sources checked on 2026-10-17:
`openai/MMMLU` (15 configs, no Albanian) and INCLUDE
(`CohereLabs/include-base-44`, config `Albanian`, 551 test rows, regional exam
questions with a different schema); INCLUDE is not supported yet.

Decision: download only the Parquet file of the selected dataset and language
on first use and cache it under `DATA_CACHE_DIR/<dataset key>/`. Rationale: a few MB per language,
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

Confidence interval: every accuracy (running, overall, per subject category,
per cultural sensitivity label) comes with the 95% Wilson score interval
(`ci_low`, `ci_high`, z = 1.96), computed in `backend/app/stats.py`. Wilson
was chosen over the normal approximation because it stays within [0, 1] and
is meaningful for small groups and accuracies near 0% or 100% (10/10 gives
72.2–100%). The interval covers sampling error only, not dataset errors or
translation quality.

Reasoning effort: optional per run (`NONE`, `MINIMAL`, `LOW`, `MEDIUM`,
`HIGH`, the values of `GenericChatRequest.reasoning_effort` in SDK 2.188.0).
When omitted the parameter is not sent and the model uses its own default.
No effort is forced on any model.

Observed on 2026-10-17 with a minimal prompt (`eu-frankfurt-1`):

| Model | Not sent | `NONE` | `LOW` / `HIGH` |
| --- | --- | --- | --- |
| `openai.gpt-4.1` | OK | HTTP 400 | HTTP 400 |
| `meta.llama-3.3-70b-instruct` | OK | ignored | ignored |
| `google.gemini-2.5-flash` | OK | OK, no reasoning tokens | OK |
| `openai.gpt-5.6-luna` | OK | OK | OK |
| dedicated gpt-oss-120b endpoint | OK | HTTP 400 | OK |
| dedicated Qwen 3.6 endpoint | long reasoning | OK, no reasoning | OK |

Because an HTTP 400 means the request itself is invalid for the model, the
evaluator stops the run at the first one with an `error` event (partial
results saved with `stopped: true`) instead of failing every question.

Diagnosis that motivated the option: on the dedicated Qwen 3.6 endpoint with
`max_tokens=2000` and no effort, 17 of 100 answers were empty, each after about
25.7 s. A replayed question returned `finish_reason=length` with all 2000
completion tokens spent on reasoning (`reasoning_content`) and no content;
with 8000 tokens the result was the same after 97 s; with `NONE` the model
answered in 0.3 s.

## Intended behavior

API (all JSON, base `http://localhost:8000`):

| Method | Path | Result |
| --- | --- | --- |
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/models` | configured model IDs and the default |
| GET | `/api/datasets` | dataset keys, names, test-split sizes, and languages |
| GET | `/api/languages?dataset=<key>` | language codes and English names of a dataset |
| GET | `/api/runs` | normalized summaries of the saved runs, newest first |
| GET | `/api/runs/{run_id}` | one saved run: summary, breakdowns, answers; 404 if absent |
| POST | `/api/evaluations` | SSE stream of a run |

`POST /api/evaluations` body: `model_id`, `language`, `num_questions`
(1 to the smaller of `MAX_QUESTIONS` and the dataset's test size), optional
`dataset` (`global-mmlu`, the default, or `global-mmlu-lite`), required `seed`
(non-negative integer), optional `reasoning_effort`. A missing seed, an
unknown model, dataset, or effort, or a language the dataset does not have,
returns 422 before any remote call. `GET /api/models` also returns the allowed
`reasoning_efforts` and `default_seed`.

Reproducibility: every run has an explicit seed; there is no random seed.
`DEFAULT_SEED` (default 42) is only the value the UI pre-fills. The sample is
`random.Random(seed).sample` over the row indices of the cached test split,
so it is stable for a given dataset revision.

SSE events (`data:` lines carrying JSON with a `type` field):

* `status`: preparation messages (for example dataset download).
* `start`: `run_id`, `seed`, `total`, `model_id`, `reasoning_effort`,
  `dataset`, `language`.
* `result`: one per question, in completion order: `index`, `sample_id`,
  `subject`, `subject_category`, `cultural_sensitivity_label`, `expected`,
  `predicted`, `correct`, `invalid`, `latency_s`, `error`, plus running
  `completed`, `correct_count`, `accuracy`, `ci_low`, and `ci_high`.
* `done`: final summary with overall accuracy, accuracy by
  `subject_category` and by cultural sensitivity label, invalid and error
  counts, mean latency, `dataset`, `reasoning_effort`, `model_target`, `stopped: false`,
  and the path of the saved result file.
* `error`: fatal error, including a request rejected with HTTP 400; the
  stream ends.

Calls run concurrently, bounded by `MAX_CONCURRENCY`. A failed call (after the
SDK retry strategy) is recorded with `error` and counted as wrong; the run
continues. If the client disconnects (Stop button), no new calls are started
and the partial results are saved with `stopped: true`.

Runs history: `backend/app/runs.py` reads `RESULTS_DIR/*.json` and maps every
saved format to the current summary fields. Files older than a feature get
`dataset: global-mmlu`, `stopped: false`, `model_target` equal to the label,
and intervals recomputed from the counts. `reasoning_effort_recorded: false`
marks runs saved before the effort was recorded (then on-demand OpenAI models
always got `LOW`), so they are not confused with "not sent". Unreadable files
are skipped and logged. Run IDs are checked against `[A-Za-z0-9._-]+` before
a file is opened.

## Configuration

Environment variables (`.env`, see `.env.example`): `OCI_CONFIG_FILE`,
`OCI_PROFILE`, `OCI_REGION`, `OCI_COMPARTMENT_ID`, `MODELS`, `DEFAULT_MODEL`,
`DEFAULT_SEED`, `MAX_QUESTIONS`, `MAX_CONCURRENCY`, `MAX_COMPLETION_TOKENS`, `DATA_CACHE_DIR`,
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
3. A seed is required; the same seed, dataset, language, and N select the
   same questions.
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

Reasoning effort, API, 2026-10-17:

| Check | Observed |
| --- | --- |
| Dedicated Qwen 3.6, `it`, N=50, seed 1911322290, effort not sent (UI run) | 38/50 (0.76), 8 invalid, mean latency 6.6 s, max 25.8 s |
| Same sample, `NONE` | 37/50 (0.74), 0 invalid, mean latency 0.26 s, max 1.5 s |
| `openai.gpt-4.1`, `LOW`, N=20 | `start` then `error` with the service's HTTP 400 message; no results |
| `openai.gpt-4.1`, not sent, N=3 | 3/3, no errors |

Dataset selection, 2026-10-17:

| Check | Observed |
| --- | --- |
| API, Lite, `sq`, N=20, seed 42, `openai.gpt-5.6-luna` | status (Lite download), 20 results, 18/20 (0.90); file in `data/cache/global-mmlu-lite/` |
| API, Global-MMLU, `sq` | 422 |
| UI, Lite, `sq`, N=10 | selector shows 23 languages and max 400; 8/10; title shows the dataset |

Confidence interval and runs page, 2026-10-18:

| Check | Observed |
| --- | --- |
| Unit tests for Wilson values (80/100, 10/10, 0/10, 200/250) | match reference values within 0.001 |
| `list_runs` on the 78 local result files (five saved formats) | 78 listed, none skipped; every detail loads |
| UI Runs page | 78 runs; filters Albanian + `low` give 11; detail shows breakdown bars with interval whiskers and the answers |
| UI evaluation, Global-MMLU `it`, N=20, seed 42 | Accuracy tile shows the interval (17 answers: 88.2%, 65.7–96.7%); shaded band in the chart narrows as answers arrive |

Mean latency was about 1.0–1.1 s per call on demand. These are single runs on small
samples: they verify the mechanism, not the model's accuracy.

Not verified: other on-demand vendors (non-OpenAI models), Cohere formats, other regions, runs near
`MAX_QUESTIONS`, behavior under service throttling (429).
