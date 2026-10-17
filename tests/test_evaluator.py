"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Tests of prompt building, answer extraction, and the evaluation loop.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import asyncio
import json

import pytest

from app.dataset import load_questions
from app.evaluator import RunConfig, build_prompt, extract_letter, run_evaluation

from conftest import FakeChatModel


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("B", "B"),
        (" c) ", None),
        ("**D**", "D"),
        ("The answer is C.", "C"),
        ("Answer: A", "A"),
        ("I don't know", None),
        ("", None),
    ],
)
def test_extract_letter(reply, expected):
    assert extract_letter(reply) == expected


def test_build_prompt_lists_options(parquet_file):
    question = load_questions(parquet_file / "global-mmlu" / "it-test.parquet")[0]
    prompt = build_prompt(question)
    assert "Question 0?" in prompt
    assert "A. a\nB. b\nC. c\nD. d" in prompt


async def _collect(model, questions, tmp_path):
    config = RunConfig(
        run_id="run1",
        model_id="m",
        language="it",
        seed=1,
        max_concurrency=3,
        results_dir=tmp_path / "results",
    )
    return [event async for event in run_evaluation(model, questions, config)]


def test_run_reports_running_accuracy_and_summary(parquet_file, tmp_path):
    questions = load_questions(parquet_file / "global-mmlu" / "it-test.parquet")[
        :4
    ]  # answers A B C D
    model = FakeChatModel(
        replies={
            "Question 0?": "A",
            "Question 1?": "B",
            "Question 2?": "x",
            "Question 3?": "D",
        },
        fail_on={"Question 3?"},
    )
    events = asyncio.run(_collect(model, questions, tmp_path))

    assert events[0]["type"] == "start" and events[0]["total"] == 4
    results = [e for e in events if e["type"] == "result"]
    assert [e["completed"] for e in results] == [1, 2, 3, 4]
    done = events[-1]
    assert done["type"] == "done"
    assert done["correct"] == 2 and done["accuracy"] == 0.5
    assert done["invalid"] == 1 and done["errors"] == 1
    saved = json.loads((tmp_path / "results" / "run1.json").read_text())
    assert [r["index"] for r in saved["results"]] == [0, 1, 2, 3]


def test_stopping_early_cancels_pending_calls(parquet_file, tmp_path):
    questions = load_questions(parquet_file / "global-mmlu" / "it-test.parquet")
    model = FakeChatModel()
    config = RunConfig(
        run_id="run2",
        model_id="m",
        language="it",
        seed=1,
        max_concurrency=1,
        results_dir=tmp_path,
    )

    async def consume_two():
        stream = run_evaluation(model, questions, config)
        seen = []
        async for event in stream:
            seen.append(event)
            if len(seen) == 3:  # start + two results
                break
        await stream.aclose()
        await asyncio.sleep(0.05)
        return seen

    asyncio.run(consume_two())
    assert model.calls < len(questions)
    saved = json.loads((tmp_path / "run2.json").read_text())
    assert saved["summary"]["stopped"] is True
    assert saved["summary"]["completed"] == 2
