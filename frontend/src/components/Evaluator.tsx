"use client";

/**
 * Main screen: run setup (model, reasoning effort, dataset, language, number
 * of questions, seed), live progress and accuracy, per-question results, and the final
 * breakdown by subject category and cultural sensitivity.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import {
  API_BASE_URL,
  DatasetInfo,
  DoneEvent,
  EvaluationEvent,
  Language,
  ResultEvent,
  StartEvent,
  Tally,
  fetchDatasets,
  fetchModels,
  streamEvaluation,
} from "@/lib/api";

type Phase = "idle" | "preparing" | "running" | "done" | "stopped" | "error";

const QUESTION_PRESETS = [10, 50, 100, 250];

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function Evaluator() {
  const [models, setModels] = useState<string[]>([]);
  const [efforts, setEfforts] = useState<string[]>([]);
  const [region, setRegion] = useState("");
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  const [modelId, setModelId] = useState("");
  const [datasetKey, setDatasetKey] = useState("");
  const [language, setLanguage] = useState("it");
  const [numQuestions, setNumQuestions] = useState(50);
  // Required for reproducible samples; pre-filled with DEFAULT_SEED from the backend.
  const [seed, setSeed] = useState("");
  const [defaultSeed, setDefaultSeed] = useState<number | null>(null);
  // Empty string: "Model default", the parameter is not sent.
  const [reasoningEffort, setReasoningEffort] = useState("");

  const [phase, setPhase] = useState<Phase>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [start, setStart] = useState<StartEvent | null>(null);
  const [results, setResults] = useState<ResultEvent[]>([]);
  const [summary, setSummary] = useState<DoneEvent | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    Promise.all([fetchModels(), fetchDatasets()])
      .then(([catalog, datasetCatalog]) => {
        setModels(catalog.models);
        setModelId(catalog.default);
        setRegion(catalog.region);
        setEfforts(catalog.reasoning_efforts ?? []);
        if (typeof catalog.default_seed === "number") {
          setDefaultSeed(catalog.default_seed);
          setSeed(String(catalog.default_seed));
        }
        setDatasets(datasetCatalog.datasets);
        setDatasetKey(datasetCatalog.default);
      })
      .catch((error: Error) =>
        setCatalogError(
          `Cannot reach the backend at ${API_BASE_URL} (${error.message}). Is it running?`,
        ),
      );
  }, []);

  const dataset = datasets.find((d) => d.key === datasetKey);
  const languages: Language[] = dataset?.languages ?? [];
  const maxQuestions = dataset?.test_rows ?? 1;

  // Keep the language if the new dataset has it, else fall back to English.
  function selectDataset(key: string) {
    setDatasetKey(key);
    const next = datasets.find((d) => d.key === key);
    if (next && !next.languages.some((l) => l.code === language)) {
      setLanguage(
        next.languages.some((l) => l.code === "en") ? "en" : next.languages[0].code,
      );
    }
    if (next && numQuestions > next.test_rows) setNumQuestions(next.test_rows);
  }

  const running = phase === "preparing" || phase === "running";
  const latest = results.at(-1);
  const total = start?.total ?? numQuestions;
  const completed = latest?.completed ?? 0;
  const accuracy = latest?.accuracy ?? 0;
  const invalid = results.filter((r) => r.invalid).length;
  const errors = results.filter((r) => r.error).length;
  const meanLatency = results.length
    ? results.reduce((sum, r) => sum + r.latency_s, 0) / results.length
    : 0;
  const languageName = (code: string) =>
    datasets.flatMap((d) => d.languages).find((l) => l.code === code)?.name ?? code;
  const datasetName = (key: string) => datasets.find((d) => d.key === key)?.name ?? key;

  function handleEvent(event: EvaluationEvent) {
    switch (event.type) {
      case "status":
        setStatusMessage(event.message);
        break;
      case "start":
        setStart(event);
        setPhase("running");
        setStatusMessage("");
        break;
      case "result":
        setResults((previous) => [...previous, event]);
        break;
      case "done":
        setSummary(event);
        setPhase("done");
        break;
      case "error":
        setRunError(event.message);
        setPhase("error");
        break;
    }
  }

  async function startRun() {
    const controller = new AbortController();
    abortRef.current = controller;
    setPhase("preparing");
    setStatusMessage("Preparing the sample…");
    setStart(null);
    setResults([]);
    setSummary(null);
    setRunError(null);
    try {
      await streamEvaluation(
        {
          model_id: modelId,
          dataset: datasetKey,
          language,
          num_questions: numQuestions,
          seed: Number(seed),
          ...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
        },
        handleEvent,
        controller.signal,
      );
      setPhase((current) => (current === "running" || current === "preparing" ? "stopped" : current));
    } catch (error) {
      if (controller.signal.aborted) {
        setPhase("stopped");
      } else {
        setRunError((error as Error).message);
        setPhase("error");
      }
    } finally {
      abortRef.current = null;
    }
  }

  function stopRun() {
    abortRef.current?.abort();
  }

  const seedValid = /^\d+$/.test(seed.trim());
  const canStart =
    !running &&
    !!modelId &&
    !!dataset &&
    !!language &&
    numQuestions >= 1 &&
    numQuestions <= maxQuestions &&
    seedValid;

  return (
    <main className="page">
      <header className="header">
        <div>
          <h1>Global-MMLU multilingual evaluation</h1>
          <p className="muted">
            Random questions from the <code>{dataset?.hf_id ?? "CohereLabs/Global-MMLU"}</code>{" "}
            test split, answered by an OCI Generative AI model
            {region ? ` in ${region}` : ""}.
          </p>
        </div>
      </header>

      {catalogError && <div className="alert">{catalogError}</div>}

      <div className="layout">
        <section className="card setup">
          <h2>Run setup</h2>
          <label>
            Model
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={running || models.length === 0}
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>

          <label>
            Reasoning effort
            <select
              value={reasoningEffort}
              onChange={(e) => setReasoningEffort(e.target.value)}
              disabled={running}
            >
              <option value="">Model default (not sent)</option>
              {efforts.map((effort) => (
                <option key={effort} value={effort}>
                  {effort.toLowerCase()}
                </option>
              ))}
            </select>
            <span className="hint">
              Some models reject this setting; the run then stops with an error.
            </span>
          </label>

          <label>
            Dataset
            <select
              value={datasetKey}
              onChange={(e) => selectDataset(e.target.value)}
              disabled={running || datasets.length === 0}
            >
              {datasets.map((d) => (
                <option key={d.key} value={d.key}>
                  {d.name} ({d.languages.length} languages)
                </option>
              ))}
            </select>
            {dataset && (
              <span className="hint">
                {dataset.hf_id} · {dataset.test_rows.toLocaleString("en-US")} test
                questions per language
              </span>
            )}
          </label>

          <label>
            Language
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={running || languages.length === 0}
            >
              {[...languages]
                .sort((a, b) => a.name.localeCompare(b.name))
                .map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.name} ({l.code})
                  </option>
                ))}
            </select>
          </label>

          <label>
            Number of questions <span className="muted">(max {maxQuestions})</span>
            <input
              type="number"
              min={1}
              max={maxQuestions}
              value={numQuestions}
              onChange={(e) => setNumQuestions(Math.max(1, Number(e.target.value) || 1))}
              disabled={running}
              aria-invalid={numQuestions > maxQuestions}
            />
          </label>
          <div className="presets">
            {QUESTION_PRESETS.filter((n) => n <= maxQuestions).map((n) => (
              <button
                key={n}
                type="button"
                className={n === numQuestions ? "chip active" : "chip"}
                onClick={() => setNumQuestions(n)}
                disabled={running}
              >
                {n}
              </button>
            ))}
          </div>

          <label>
            Seed <span className="muted">(required, same seed = same questions)</span>
            <input
              type="text"
              inputMode="numeric"
              required
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              disabled={running}
              aria-invalid={!seedValid}
            />
            {defaultSeed !== null && (
              <span className="hint">
                Default from .env: {defaultSeed}.{" "}
                {seed.trim() !== String(defaultSeed) && (
                  <button
                    type="button"
                    className="link"
                    onClick={() => setSeed(String(defaultSeed))}
                    disabled={running}
                  >
                    Reset
                  </button>
                )}
              </span>
            )}
          </label>

          {running ? (
            <button type="button" className="button stop" onClick={stopRun}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="button"
              onClick={startRun}
              disabled={!canStart}
            >
              Start evaluation
            </button>
          )}
        </section>

        <section className="results">
          <div className="card">
            <div className="run-title">
              <h2>
                {start
                  ? `${start.model_id} · ${datasetName(start.dataset)} · ${languageName(start.language)}`
                  : "Progress"}
              </h2>
              <span className={`badge ${phase}`}>{phase}</span>
            </div>
            {start && (
              <p className="muted">
                Reasoning effort {start.reasoning_effort?.toLowerCase() ?? "model default"} ·
                seed {start.seed} · run {start.run_id}
              </p>
            )}
            {statusMessage && <p className="muted">{statusMessage}</p>}
            {runError && <div className="alert">{runError}</div>}

            <div className="progress" aria-label="progress">
              <div
                className="progress-fill"
                style={{ width: `${total ? (completed / total) * 100 : 0}%` }}
              />
            </div>

            <div className="tiles">
              <Tile label="Answered" value={`${completed} / ${total}`} />
              <Tile label="Accuracy" value={completed ? percent(accuracy) : "—"} strong />
              <Tile label="Correct" value={String(latest?.correct_count ?? 0)} />
              <Tile label="Invalid · errors" value={`${invalid} · ${errors}`} />
              <Tile
                label="Mean latency"
                value={completed ? `${meanLatency.toFixed(2)} s` : "—"}
              />
            </div>

            <AccuracyChart results={results} />
          </div>

          {summary && (
            <div className="card">
              <h2>Breakdown</h2>
              <div className="breakdowns">
                <Breakdown title="Subject category" data={summary.by_subject_category} />
                <Breakdown
                  title="Cultural sensitivity"
                  data={summary.by_cultural_sensitivity}
                  labels={{ CS: "Culturally sensitive", CA: "Culturally agnostic", "-": "Not annotated" }}
                />
              </div>
              <p className="muted small">Saved to {summary.result_file}</p>
            </div>
          )}

          {results.length > 0 && <ResultsTable results={results} />}
        </section>
      </div>
    </main>
  );
}

function Tile({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className={strong ? "tile strong" : "tile"}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
    </div>
  );
}

/** Running accuracy after each answer, as a line on a 0–100% scale. */
function AccuracyChart({ results }: { results: ResultEvent[] }) {
  const width = 600;
  const height = 120;
  const pad = 4;
  const points = useMemo(() => {
    if (results.length < 2) return "";
    return results
      .map((r, i) => {
        const x = (i / (results.length - 1)) * width;
        // Keep 0% and 100% inside the plot area, so the line stays visible.
        const y = pad + (1 - r.accuracy) * (height - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [results]);

  return (
    <div className="chart">
      <div className="chart-label muted small">Running accuracy</div>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img"
        aria-label="Running accuracy after each answer">
        {[0.25, 0.5, 0.75].map((level) => (
          <line key={level} x1={0} x2={width} y1={height - level * height}
            y2={height - level * height} className="gridline" />
        ))}
        {points && <polyline points={points} className="line" />}
      </svg>
    </div>
  );
}

function Breakdown({
  title,
  data,
  labels = {},
}: {
  title: string;
  data: Record<string, Tally>;
  labels?: Record<string, string>;
}) {
  return (
    <div>
      <h3>{title}</h3>
      {Object.entries(data).map(([key, tally]) => (
        <div key={key} className="bar-row">
          <span className="bar-name">{labels[key] ?? key}</span>
          <div className="bar">
            <div className="bar-fill" style={{ width: `${tally.accuracy * 100}%` }} />
          </div>
          <span className="bar-value">
            {percent(tally.accuracy)} <span className="muted">({tally.correct}/{tally.total})</span>
          </span>
        </div>
      ))}
    </div>
  );
}

function ResultsTable({ results }: { results: ResultEvent[] }) {
  const rows = [...results].reverse();
  return (
    <div className="card">
      <h2>Answers</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Subject</th>
              <th>Category</th>
              <th>CS/CA</th>
              <th>Expected</th>
              <th>Predicted</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.index} className={r.correct ? "ok" : "ko"}>
                <td>{r.index + 1}</td>
                <td title={r.sample_id}>{r.subject.replaceAll("_", " ")}</td>
                <td>{r.subject_category}</td>
                <td>{r.cultural_sensitivity_label}</td>
                <td>{r.expected}</td>
                <td title={r.error ?? r.reply}>
                  {r.error ? "error" : (r.predicted ?? "invalid")}
                </td>
                <td>{r.latency_s.toFixed(2)} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
