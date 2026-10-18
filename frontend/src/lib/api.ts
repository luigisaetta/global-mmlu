/**
 * Client for the local FastAPI backend: catalog endpoints and the SSE
 * evaluation stream (POST, so it is read with fetch rather than EventSource).
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface Language {
  code: string;
  name: string;
}

export interface DatasetInfo {
  key: string;
  name: string;
  hf_id: string;
  test_rows: number;
  languages: Language[];
}

export interface ModelCatalog {
  models: string[];
  default: string;
  region: string;
  reasoning_efforts: string[];
  default_seed: number;
}

export interface Tally {
  total: number;
  correct: number;
  accuracy: number;
  /** 95% Wilson interval; absent in runs saved before it was added. */
  ci_low?: number;
  ci_high?: number;
}

/** One graded answer, as streamed and as saved in a result file. */
export interface AnswerRecord {
  index: number;
  sample_id: string;
  subject: string;
  subject_category: string;
  cultural_sensitivity_label: string;
  expected: string;
  predicted: string | null;
  correct: boolean;
  invalid: boolean;
  latency_s: number;
  error: string | null;
  reply: string;
}

export interface ResultEvent extends AnswerRecord {
  type: "result";
  completed: number;
  correct_count: number;
  accuracy: number;
  ci_low: number;
  ci_high: number;
}

export interface StartEvent {
  type: "start";
  run_id: string;
  model_id: string;
  reasoning_effort: string | null;
  dataset: string;
  language: string;
  seed: number;
  total: number;
}

export interface DoneEvent extends Tally {
  type: "done";
  run_id: string;
  reasoning_effort: string | null;
  dataset: string;
  seed: number;
  requested: number;
  completed: number;
  by_subject_category: Record<string, Tally>;
  by_cultural_sensitivity: Record<string, Tally>;
  invalid: number;
  errors: number;
  mean_latency_s: number;
  result_file: string;
}

export type EvaluationEvent =
  | StartEvent
  | ResultEvent
  | DoneEvent
  | { type: "status"; message: string }
  | { type: "error"; message: string };

export interface EvaluationRequest {
  model_id: string;
  dataset: string;
  language: string;
  num_questions: number;
  seed: number;
  /** Omitted: the model uses its own default reasoning effort. */
  reasoning_effort?: string;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`${path} returned HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const fetchModels = () => getJson<ModelCatalog>("/api/models");

/** Normalized summary of a saved run (GET /api/runs). */
export interface RunSummary {
  run_id: string;
  saved_at: string | null;
  model_id: string;
  model_target: string;
  dataset: string;
  language: string;
  /** null: not sent (model default), when reasoning_effort_recorded is true. */
  reasoning_effort: string | null;
  /** false for runs saved before the effort was recorded. */
  reasoning_effort_recorded: boolean;
  seed: number | null;
  requested: number;
  completed: number;
  correct: number;
  accuracy: number;
  ci_low: number;
  ci_high: number;
  invalid: number;
  errors: number;
  mean_latency_s: number;
  stopped: boolean;
}

export interface RunDetail {
  summary: RunSummary;
  by_subject_category: Record<string, Tally>;
  by_cultural_sensitivity: Record<string, Tally>;
  results: AnswerRecord[];
}

export const fetchRuns = () =>
  getJson<{ runs: RunSummary[] }>("/api/runs").then((r) => r.runs);

export const fetchRun = (runId: string) =>
  getJson<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);

export const fetchDatasets = () =>
  getJson<{ default: string; datasets: DatasetInfo[] }>("/api/datasets");

/**
 * Start an evaluation and call `onEvent` for every SSE message until the
 * stream ends or `signal` aborts it. Throws on HTTP errors with the backend's
 * `detail` message.
 */
export async function streamEvaluation(
  request: EvaluationRequest,
  onEvent: (event: EvaluationEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/evaluations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    const detail =
      typeof body?.detail === "string" ? body.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const message = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = message
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data) as EvaluationEvent);
      boundary = buffer.indexOf("\n\n");
    }
  }
}
