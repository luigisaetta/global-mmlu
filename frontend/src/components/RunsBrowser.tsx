"use client";

/**
 * Runs history: the result files saved by the backend, filterable by model,
 * dataset, language, and reasoning effort, with accuracy and 95% confidence
 * interval, sorted by date, by accuracy, or by the lower bound of the
 * interval (which favors large, reliable runs). Stopped runs are hidden unless
 * "Show stopped runs" is checked.
 * Selecting a run shows its breakdowns and answers.
 */

import { useEffect, useMemo, useState } from "react";

import {
  API_BASE_URL,
  DatasetInfo,
  RunDetail,
  RunSummary,
  fetchDatasets,
  fetchRun,
  fetchRuns,
} from "@/lib/api";
import { formatCi, percent } from "@/lib/format";
import { Breakdown, ResultsTable, SENSITIVITY_LABELS } from "@/components/RunViews";

const ALL = "__all";
// Effort filter values for runs without an explicit effort.
const EFFORT_DEFAULT = "__default";
const EFFORT_UNRECORDED = "__unrecorded";

type Filters = { model: string; dataset: string; language: string; effort: string };

const INITIAL_FILTERS: Filters = { model: ALL, dataset: ALL, language: ALL, effort: ALL };

/** Effort filter key of a run: its value, "model default", or "not recorded". */
function effortKey(run: RunSummary): string {
  if (!run.reasoning_effort_recorded) return EFFORT_UNRECORDED;
  return run.reasoning_effort ?? EFFORT_DEFAULT;
}

function effortLabel(key: string): string {
  if (key === EFFORT_DEFAULT) return "model default";
  if (key === EFFORT_UNRECORDED) return "not recorded";
  return key.toLowerCase();
}

const formatDate = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString("en-GB", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";

const uniqueSorted = (values: string[]) => [...new Set(values)].sort();

type SortOrder = "newest" | "accuracy" | "ciLow";

const SORT_LABELS: Record<SortOrder, string> = {
  newest: "Newest first",
  accuracy: "Accuracy (high to low)",
  ciLow: "Lower 95% bound (high to low)",
};

const byNewest = (a: RunSummary, b: RunSummary) =>
  (b.saved_at ?? "").localeCompare(a.saved_at ?? "");

/**
 * Sort runs. "ciLow" ranks by the lower bound of the 95% interval: a small
 * run at 100% (3/3 gives 43.9–100%) falls below a large run at 95%, so the
 * top of the list is what the data supports. Ties go to the higher accuracy,
 * then to the newer run.
 */
function sortRuns(runs: RunSummary[], order: SortOrder): RunSummary[] {
  const sorted = [...runs];
  if (order === "accuracy") {
    sorted.sort((a, b) => b.accuracy - a.accuracy || byNewest(a, b));
  } else if (order === "ciLow") {
    sorted.sort(
      (a, b) => b.ci_low - a.ci_low || b.accuracy - a.accuracy || byNewest(a, b),
    );
  } else {
    sorted.sort(byNewest);
  }
  return sorted;
}

export default function RunsBrowser() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<Filters>(INITIAL_FILTERS);
  // Stopped runs are partial; they stay on disk but are hidden by default.
  const [showStopped, setShowStopped] = useState(false);
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Fetches runs and dataset names; state is only set in the callbacks.
  function fetchAll() {
    return Promise.all([fetchRuns(), fetchDatasets()])
      .then(([runList, datasetCatalog]) => {
        setRuns(runList);
        setDatasets(datasetCatalog.datasets);
        setLoadError(null);
      })
      .catch((error: Error) =>
        setLoadError(
          `Cannot load runs from ${API_BASE_URL} (${error.message}). Is the backend running?`,
        ),
      )
      .finally(() => setLoading(false));
  }

  function refresh() {
    setLoading(true);
    fetchAll();
  }

  useEffect(() => {
    fetchAll();
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    fetchRun(selectedId)
      .then((run) => {
        if (!cancelled) setDetail(run);
      })
      .catch((error: Error) => {
        if (!cancelled) setDetailError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  function selectRun(runId: string) {
    setDetail(null);
    setDetailError(null);
    setSelectedId(runId === selectedId ? null : runId);
  }

  const languageName = (code: string) =>
    datasets.flatMap((d) => d.languages).find((l) => l.code === code)?.name ?? code;
  const datasetName = (key: string) => datasets.find((d) => d.key === key)?.name ?? key;

  // Options come from the runs themselves, so only values that exist are offered.
  const options = useMemo(
    () => ({
      model: uniqueSorted(runs.map((r) => r.model_id)),
      dataset: uniqueSorted(runs.map((r) => r.dataset)),
      language: uniqueSorted(runs.map((r) => r.language)),
      effort: uniqueSorted(runs.map(effortKey)),
    }),
    [runs],
  );

  const stoppedCount = runs.filter((r) => r.stopped).length;
  const visible = sortRuns(
    runs.filter(
      (r) =>
        (showStopped || !r.stopped) &&
        (filters.model === ALL || r.model_id === filters.model) &&
        (filters.dataset === ALL || r.dataset === filters.dataset) &&
        (filters.language === ALL || r.language === filters.language) &&
        (filters.effort === ALL || effortKey(r) === filters.effort),
    ),
    sortOrder,
  );

  const setFilter = (name: keyof Filters) => (value: string) =>
    setFilters((current) => ({ ...current, [name]: value }));
  const filtersActive = Object.values(filters).some((value) => value !== ALL);

  return (
    <main className="page">
      <header className="header">
        <h1>Runs</h1>
        <p className="muted">
          Saved evaluations from the backend&apos;s <code>results/</code> folder, sorted
          by date, accuracy, or lower bound of the 95% Wilson confidence interval.
        </p>
      </header>

      {loadError && <div className="alert">{loadError}</div>}

      <section className="card">
        <div className="filters">
          <FilterSelect
            label="Model"
            value={filters.model}
            options={options.model}
            onChange={setFilter("model")}
          />
          <FilterSelect
            label="Dataset"
            value={filters.dataset}
            options={options.dataset}
            format={datasetName}
            onChange={setFilter("dataset")}
          />
          <FilterSelect
            label="Language"
            value={filters.language}
            options={options.language}
            format={(code) => `${languageName(code)} (${code})`}
            onChange={setFilter("language")}
          />
          <FilterSelect
            label="Reasoning effort"
            value={filters.effort}
            options={options.effort}
            format={effortLabel}
            onChange={setFilter("effort")}
          />
          <label className="filter">
            Sort by
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value as SortOrder)}
            >
              {(Object.keys(SORT_LABELS) as SortOrder[]).map((order) => (
                <option key={order} value={order}>
                  {SORT_LABELS[order]}
                </option>
              ))}
            </select>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={showStopped}
              onChange={(e) => setShowStopped(e.target.checked)}
            />
            Show stopped runs ({stoppedCount})
          </label>
          <div className="filter-actions">
            <button
              type="button"
              className="chip"
              onClick={() => setFilters(INITIAL_FILTERS)}
              disabled={!filtersActive}
            >
              Clear filters
            </button>
            <button type="button" className="chip" onClick={refresh} disabled={loading}>
              {loading ? "Loading…" : "Refresh"}
            </button>
          </div>
        </div>
        <p className="muted small">
          Showing {visible.length} of {runs.length} runs
          {!showStopped && stoppedCount > 0 && ` (${stoppedCount} stopped hidden)`}.
        </p>

        <div className="table-wrap runs-table">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>Language</th>
                <th>Effort</th>
                <th>Seed</th>
                <th>Answered</th>
                <th>Accuracy</th>
                <th>95% CI</th>
                <th>Invalid · err</th>
                <th>Latency</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr
                  key={r.run_id}
                  className={r.run_id === selectedId ? "selectable selected" : "selectable"}
                  onClick={() => selectRun(r.run_id)}
                  title={r.run_id}
                >
                  <td>{formatDate(r.saved_at)}</td>
                  <td>{r.model_id}</td>
                  <td>{datasetName(r.dataset)}</td>
                  <td>{languageName(r.language)}</td>
                  <td>{effortLabel(effortKey(r))}</td>
                  <td>{r.seed ?? "—"}</td>
                  <td>
                    {r.completed}
                    {r.completed !== r.requested && ` / ${r.requested}`}
                  </td>
                  <td className="strong-cell">{percent(r.accuracy)}</td>
                  <td>{formatCi(r.ci_low, r.ci_high)}</td>
                  <td>
                    {r.invalid} · {r.errors}
                  </td>
                  <td>{r.mean_latency_s.toFixed(2)} s</td>
                  <td>
                    <span className={`badge ${r.stopped ? "stopped" : "done"}`}>
                      {r.stopped ? "stopped" : "done"}
                    </span>
                  </td>
                </tr>
              ))}
              {!loading && visible.length === 0 && (
                <tr>
                  <td colSpan={12} className="muted">
                    No runs match the filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {selectedId && (
        <section className="results run-detail">
          {detailError && <div className="alert">{detailError}</div>}
          {!detail && !detailError && <p className="muted">Loading run…</p>}
          {detail && (
            <>
              <div className="card">
                <h2>
                  {detail.summary.model_id} · {datasetName(detail.summary.dataset)} ·{" "}
                  {languageName(detail.summary.language)}
                </h2>
                <p className="muted small">
                  {percent(detail.summary.accuracy)} ({detail.summary.correct}/
                  {detail.summary.completed}), 95% CI{" "}
                  {formatCi(detail.summary.ci_low, detail.summary.ci_high)} · seed{" "}
                  {detail.summary.seed ?? "—"} · effort{" "}
                  {effortLabel(effortKey(detail.summary))} · run {detail.summary.run_id}
                  {detail.summary.model_target !== detail.summary.model_id &&
                    " · dedicated endpoint"}
                </p>
                <div className="breakdowns">
                  <Breakdown title="Subject category" data={detail.by_subject_category} />
                  <Breakdown
                    title="Cultural sensitivity"
                    data={detail.by_cultural_sensitivity}
                    labels={SENSITIVITY_LABELS}
                  />
                </div>
              </div>
              {detail.results.length > 0 && (
                <ResultsTable results={detail.results} newestFirst={false} />
              )}
            </>
          )}
        </section>
      )}
    </main>
  );
}

function FilterSelect({
  label,
  value,
  options,
  format = (option) => option,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  format?: (option: string) => string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="filter">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value={ALL}>All</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {format(option)}
          </option>
        ))}
      </select>
    </label>
  );
}
