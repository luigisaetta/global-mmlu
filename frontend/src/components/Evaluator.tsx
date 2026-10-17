"use client";

/**
 * Main screen: run setup (model, language, number of questions, optional
 * seed), live progress and accuracy, per-question results, and the final
 * breakdown by subject category and cultural sensitivity.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import {
  API_BASE_URL,
  DoneEvent,
  EvaluationEvent,
  Language,
  ResultEvent,
  StartEvent,
  Tally,
  fetchLanguages,
  fetchModels,
  streamEvaluation,
} from "@/lib/api";

type Phase = "idle" | "preparing" | "running" | "done" | "stopped" | "error";

const QUESTION_PRESETS = [10, 50, 100, 250];

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function Evaluator() {
  const [models, setModels] = useState<string[]>([]);
  const [region, setRegion] = useState("");
  const [languages, setLanguages] = useState<Language[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  const [modelId, setModelId] = useState("");
  const [language, setLanguage] = useState("it");
  const [numQuestions, setNumQuestions] = useState(50);
  const [seed, setSeed] = useState("");

  const [phase, setPhase] = useState<Phase>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [start, setStart] = useState<StartEvent | null>(null);
  const [results, setResults] = useState<ResultEvent[]>([]);
  const [summary, setSummary] = useState<DoneEvent | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    Promise.all([fetchModels(), fetchLanguages()])
      .then(([catalog, langs]) => {
        setModels(catalog.models);
        setModelId(catalog.default);
        setRegion(catalog.region);
        setLanguages(langs);
      })
      .catch((error: Error) =>
        setCatalogError(
          `Cannot reach the backend at ${API_BASE_URL} (${error.message}). Is it running?`,
        ),
      );
  }, []);

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
    languages.find((l) => l.code === code)?.name ?? code;

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
          language,
          num_questions: numQuestions,
          ...(seed.trim() ? { seed: Number(seed) } : {}),
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

  const seedValid = seed.trim() === "" || /^\d+$/.test(seed.trim());
  const canStart =
    !running && !!modelId && !!language && numQuestions >= 1 && seedValid;

  return (
    <main className="page">
      <header className="header">
        <div>
          <h1>Global-MMLU multilingual evaluation</h1>
          <p className="muted">
            Random questions from the <code>CohereLabs/Global-MMLU</code> test split,
            answered by an OCI Generative AI model
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
            Number of questions
            <input
              type="number"
              min={1}
              value={numQuestions}
              onChange={(e) => setNumQuestions(Math.max(1, Number(e.target.value) || 1))}
              disabled={running}
            />
          </label>
          <div className="presets">
            {QUESTION_PRESETS.map((n) => (
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
            Seed <span className="muted">(optional, for a reproducible sample)</span>
            <input
              type="text"
              inputMode="numeric"
              placeholder="random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              disabled={running}
              aria-invalid={!seedValid}
            />
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
                  ? `${start.model_id} · ${languageName(start.language)}`
                  : "Progress"}
              </h2>
              <span className={`badge ${phase}`}>{phase}</span>
            </div>
            {start && <p className="muted">Seed {start.seed} · run {start.run_id}</p>}
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
