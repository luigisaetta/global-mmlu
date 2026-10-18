"""
Author: L. Saetta
Date last modified: 2026-10-18
License: MIT
Description: Tests of the runs history: normalization of old files and the runs API.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import json

import pytest

from app.runs import RunNotFoundError, list_runs, load_run

OLD_RUN = {  # first file format: no dataset, effort, stopped, target, or interval
    "saved_at": "2026-10-17T08:47:48+00:00",
    "summary": {
        "run_id": "old-run",
        "model_id": "openai.gpt-5.6-luna",
        "language": "it",
        "seed": 42,
        "requested": 5,
        "completed": 5,
        "total": 5,
        "correct": 4,
        "accuracy": 0.8,
        "by_subject_category": {"STEM": {"total": 5, "correct": 4, "accuracy": 0.8}},
        "by_cultural_sensitivity": {},
        "invalid": 0,
        "errors": 0,
        "mean_latency_s": 1.0,
    },
    "results": [{"index": 0}],
}

NEW_RUN = {
    "saved_at": "2026-10-18T10:00:00+00:00",
    "summary": {
        **OLD_RUN["summary"],
        "run_id": "new-run",
        "dataset": "global-mmlu-lite",
        "language": "sq",
        "reasoning_effort": None,
        "model_target": "openai.gpt-5.6-luna",
        "stopped": True,
    },
    "results": [],
}


@pytest.fixture
def results_dir(tmp_path):
    for name, payload in (("old-run", OLD_RUN), ("new-run", NEW_RUN)):
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    (tmp_path / "broken.json").write_text("{not json")
    return tmp_path


def test_list_normalizes_old_files_and_sorts_newest_first(results_dir):
    runs = list_runs(results_dir)
    assert [r["run_id"] for r in runs] == ["new-run", "old-run"]  # broken skipped
    old = runs[1]
    assert old["dataset"] == "global-mmlu"
    assert old["reasoning_effort_recorded"] is False
    assert old["stopped"] is False
    assert old["ci_low"] < 0.8 < old["ci_high"]
    new = runs[0]
    assert new["reasoning_effort"] is None and new["reasoning_effort_recorded"] is True
    assert new["stopped"] is True


def test_missing_results_dir_gives_empty_list(tmp_path):
    assert not list_runs(tmp_path / "absent")


def test_load_run_adds_intervals_to_breakdowns(results_dir):
    run = load_run(results_dir, "old-run")
    stem = run["by_subject_category"]["STEM"]
    assert stem["ci_low"] < stem["accuracy"] < stem["ci_high"]
    assert run["results"] == [{"index": 0}]


@pytest.mark.parametrize("run_id", ["absent", "../old-run", "old-run/../x"])
def test_load_run_rejects_unknown_and_unsafe_ids(results_dir, run_id):
    with pytest.raises(RunNotFoundError):
        load_run(results_dir, run_id)
