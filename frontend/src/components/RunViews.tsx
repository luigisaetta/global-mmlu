/**
 * Presentational components shared by the evaluation page and the runs page:
 * stat tile, accuracy breakdown with 95% confidence interval, answers table.
 */

import { AnswerRecord, Tally } from "@/lib/api";
import { formatCi, percent } from "@/lib/format";

export const SENSITIVITY_LABELS: Record<string, string> = {
  CS: "Culturally sensitive",
  CA: "Culturally agnostic",
  "-": "Not annotated",
};

export function Tile({
  label,
  value,
  detail,
  strong,
}: {
  label: string;
  value: string;
  detail?: string;
  strong?: boolean;
}) {
  return (
    <div className={strong ? "tile strong" : "tile"}>
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {detail && <div className="tile-detail">{detail}</div>}
    </div>
  );
}

/**
 * Accuracy per group as a bar, with the 95% Wilson interval drawn as a band
 * and written as text. Runs saved before the interval existed have no band.
 */
export function Breakdown({
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
      {Object.entries(data).map(([key, tally]) => {
        const hasCi = tally.ci_low !== undefined && tally.ci_high !== undefined;
        return (
          <div key={key} className="bar-row">
            <span className="bar-name">{labels[key] ?? key}</span>
            <div
              className="bar"
              title={hasCi ? `95% CI ${formatCi(tally.ci_low!, tally.ci_high!)}` : undefined}
            >
              <div className="bar-fill" style={{ width: `${tally.accuracy * 100}%` }} />
              {hasCi && (
                <div
                  className="bar-ci"
                  style={{
                    left: `${tally.ci_low! * 100}%`,
                    width: `${(tally.ci_high! - tally.ci_low!) * 100}%`,
                  }}
                />
              )}
            </div>
            <span className="bar-value">
              {percent(tally.accuracy)}{" "}
              <span className="muted">
                ({tally.correct}/{tally.total})
                {hasCi && ` · ${formatCi(tally.ci_low!, tally.ci_high!)}`}
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function ResultsTable({
  results,
  newestFirst = true,
}: {
  results: AnswerRecord[];
  newestFirst?: boolean;
}) {
  const rows = newestFirst ? [...results].reverse() : results;
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
