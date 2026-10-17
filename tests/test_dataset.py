"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Tests of dataset caching, loading, and seeded sampling.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import httpx
import pytest

from app.dataset import (
    DatasetError,
    ensure_language_file,
    load_questions,
    sample_questions,
)


def test_load_questions_reads_all_rows(parquet_file):
    questions = load_questions(parquet_file / "it-test.parquet")
    assert len(questions) == 40
    assert questions[1].answer == "B"
    assert questions[0].options == ("a", "b", "c", "d")


def test_sampling_is_reproducible_and_distinct(parquet_file):
    questions = load_questions(parquet_file / "it-test.parquet")
    first = sample_questions(questions, 10, seed=7)
    second = sample_questions(questions, 10, seed=7)
    other = sample_questions(questions, 10, seed=8)
    assert [q.sample_id for q in first] == [q.sample_id for q in second]
    assert [q.sample_id for q in first] != [q.sample_id for q in other]
    assert len({q.sample_id for q in first}) == 10


@pytest.mark.parametrize("count", [0, 41])
def test_sampling_rejects_out_of_range(parquet_file, count):
    questions = load_questions(parquet_file / "it-test.parquet")
    with pytest.raises(ValueError):
        sample_questions(questions, count, seed=1)


def test_cached_file_is_not_downloaded_again(parquet_file):
    def fail(_request):
        raise AssertionError("no HTTP call expected")

    client = httpx.Client(transport=httpx.MockTransport(fail))
    path = ensure_language_file("it", parquet_file, client=client)
    assert path.name == "it-test.parquet"


def test_download_writes_file_atomically(tmp_path):
    def handler(request):
        if request.url.path.endswith("/parquet/de"):
            return httpx.Response(
                200, json={"test": ["https://hf.test/de/test/0.parquet"]}
            )
        return httpx.Response(200, content=b"PARQUETDATA")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = ensure_language_file("de", tmp_path, client=client)
    assert path.read_bytes() == b"PARQUETDATA"
    assert not list(tmp_path.glob("*.part"))


def test_download_failure_raises_dataset_error(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(DatasetError):
        ensure_language_file("de", tmp_path, client=client)
    assert not list(tmp_path.iterdir())


def test_unknown_language_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ensure_language_file("xx", tmp_path)
