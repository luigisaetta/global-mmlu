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

export interface ModelCatalog {
  models: string[];
  default: string;
  region: string;
  reasoning_efforts: string[];
}

export interface Tally {
  total: number;
  correct: number;
  accuracy: number;
}

export interface ResultEvent {
  type: "result";
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
  completed: number;
  correct_count: number;
  accuracy: number;
}

export interface StartEvent {
  type: "start";
  run_id: string;
  model_id: string;
  reasoning_effort: string | null;
  language: string;
  seed: number;
  total: number;
}

export interface DoneEvent extends Tally {
  type: "done";
  run_id: string;
  reasoning_effort: string | null;
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
  language: string;
  num_questions: number;
  seed?: number;
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

export const fetchLanguages = () =>
  getJson<{ languages: Language[] }>("/api/languages").then((r) => r.languages);

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
