"""
Author: L. Saetta
Date last modified: 2026-10-18
License: MIT
Description: Read saved run files from RESULTS_DIR as normalized summaries and details.
"""

import json
import logging
import re
from pathlib import Path

from .stats import wilson_interval

logger = logging.getLogger(__name__)

# Run IDs are file names built from [A-Za-z0-9._-] (see main._SAFE_NAME_RE).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Datasets were added after the first runs, which all used Global-MMLU.
LEGACY_DATASET = "global-mmlu"


class RunNotFoundError(LookupError):
    """Raised when a run ID is malformed or has no result file."""


def _with_ci(tally: dict) -> dict:
    """Return a tally with its 95% Wilson interval (recomputed for old files)."""
    low, high = wilson_interval(tally["correct"], tally["total"])
    return {**tally, "ci_low": low, "ci_high": high}


def _normalize_summary(payload: dict, run_id: str) -> dict:
    """Map any saved summary version to the current fields.

    Older files lack ``dataset``, ``reasoning_effort``, ``model_target``,
    ``stopped``, and the confidence interval. ``reasoning_effort_recorded``
    tells "not sent" (None, recorded) apart from "not recorded" (old runs,
    when OpenAI models silently got ``LOW``).
    """
    summary = payload["summary"]
    completed = summary.get("completed", summary.get("total", 0))
    correct = summary.get("correct", 0)
    low, high = wilson_interval(correct, completed)
    return {
        "run_id": summary.get("run_id", run_id),
        "saved_at": payload.get("saved_at"),
        "model_id": summary["model_id"],
        "model_target": summary.get("model_target", summary["model_id"]),
        "dataset": summary.get("dataset", LEGACY_DATASET),
        "language": summary["language"],
        "reasoning_effort": summary.get("reasoning_effort"),
        "reasoning_effort_recorded": "reasoning_effort" in summary,
        "seed": summary.get("seed"),
        "requested": summary.get("requested", completed),
        "completed": completed,
        "correct": correct,
        "accuracy": correct / completed if completed else 0.0,
        "ci_low": low,
        "ci_high": high,
        "invalid": summary.get("invalid", 0),
        "errors": summary.get("errors", 0),
        "mean_latency_s": summary.get("mean_latency_s", 0.0),
        "stopped": bool(summary.get("stopped", False)),
    }


def _read(path: Path) -> dict:
    """Load one result file as JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(results_dir: Path) -> list[dict]:
    """Return the normalized summaries of all saved runs, newest first.

    Unreadable or malformed files are skipped and logged, so one bad file
    does not hide the others.
    """
    if not results_dir.is_dir():
        return []
    runs = []
    for path in results_dir.glob("*.json"):
        try:
            runs.append(_normalize_summary(_read(path), path.stem))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Skipping result file %s: %s", path.name, exc)
    return sorted(runs, key=lambda r: r["saved_at"] or "", reverse=True)


def load_run(results_dir: Path, run_id: str) -> dict:
    """Return one run: normalized summary, breakdowns with intervals, and answers.

    Raises:
        RunNotFoundError: if the ID is malformed or the file does not exist.
    """
    path = results_dir / f"{run_id}.json"
    if not _RUN_ID_RE.match(run_id) or not path.is_file():
        raise RunNotFoundError(run_id)
    payload = _read(path)
    summary = payload["summary"]
    return {
        "summary": _normalize_summary(payload, run_id),
        "by_subject_category": {
            k: _with_ci(v) for k, v in summary.get("by_subject_category", {}).items()
        },
        "by_cultural_sensitivity": {
            k: _with_ci(v)
            for k, v in summary.get("by_cultural_sensitivity", {}).items()
        },
        "results": payload.get("results", []),
    }
