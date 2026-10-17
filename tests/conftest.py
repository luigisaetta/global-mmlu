"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Shared fixtures: a small local Parquet file and a fake chat model.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.dataset import COLUMNS
from app.llm_client import InvalidRequestError


def make_rows(count: int) -> list[dict]:
    """Return synthetic Global-MMLU rows; the right answer cycles A-D."""
    rows = []
    for i in range(count):
        rows.append(
            {
                "sample_id": f"subject_{i % 3}/test/{i}",
                "subject": f"subject_{i % 3}",
                "subject_category": "STEM" if i % 2 else "Humanities",
                "question": f"Question {i}?",
                "option_a": "a",
                "option_b": "b",
                "option_c": "c",
                "option_d": "d",
                "answer": "ABCD"[i % 4],
                "cultural_sensitivity_label": "CS" if i % 5 == 0 else "-",
            }
        )
    return rows


@pytest.fixture
def parquet_file(tmp_path):
    """Write 40 synthetic rows to an Italian cache file and return the cache dir."""
    rows = make_rows(40)
    table = pa.table({col: [r[col] for r in rows] for col in COLUMNS})
    pq.write_table(table, tmp_path / "it-test.parquet")
    return tmp_path


class FakeChatModel:
    """Answers from a mapping of question text to reply; can fail on demand."""

    def __init__(self, replies=None, fail_on=None, reject=False):
        self.replies = replies or {}
        self.fail_on = fail_on or set()
        self.calls = 0
        self.model_ids = []
        self.efforts = []
        self.reject = reject

    def complete(self, model_id: str, prompt: str, reasoning_effort=None) -> str:
        """Return the configured reply for the question found in the prompt."""
        self.calls += 1
        self.model_ids.append(model_id)
        self.efforts.append(reasoning_effort)
        if self.reject:
            raise InvalidRequestError("HTTP 400: unsupported argument")
        for question, reply in self.replies.items():
            if question in prompt:
                if question in self.fail_on:
                    raise RuntimeError("simulated failure")
                return reply
        return "A"
