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
    DATASETS,
    DatasetError,
    ensure_language_file,
    load_questions,
    sample_questions,
)

GLOBAL = DATASETS["global-mmlu"]
LITE = DATASETS["global-mmlu-lite"]


def test_load_questions_reads_all_rows(parquet_file):
    questions = load_questions(parquet_file / "global-mmlu" / "it-test.parquet")
    assert len(questions) == 40
    assert questions[1].answer == "B"
    assert questions[0].options == ("a", "b", "c", "d")


def test_sampling_is_reproducible_and_distinct(parquet_file):
    questions = load_questions(parquet_file / "global-mmlu" / "it-test.parquet")
    first = sample_questions(questions, 10, seed=7)
    second = sample_questions(questions, 10, seed=7)
    other = sample_questions(questions, 10, seed=8)
    assert [q.sample_id for q in first] == [q.sample_id for q in second]
    assert [q.sample_id for q in first] != [q.sample_id for q in other]
    assert len({q.sample_id for q in first}) == 10


@pytest.mark.parametrize("count", [0, 41])
def test_sampling_rejects_out_of_range(parquet_file, count):
    questions = load_questions(parquet_file / "global-mmlu" / "it-test.parquet")
    with pytest.raises(ValueError):
        sample_questions(questions, count, seed=1)


def test_cached_file_is_not_downloaded_again(parquet_file):
    def fail(_request):
        raise AssertionError("no HTTP call expected")

    client = httpx.Client(transport=httpx.MockTransport(fail))
    path = ensure_language_file(GLOBAL, "it", parquet_file, client=client)
    assert path == parquet_file / "global-mmlu" / "it-test.parquet"


def test_download_writes_file_atomically(tmp_path):
    def handler(request):
        if request.url.path.endswith("/parquet/de"):
            return httpx.Response(
                200, json={"test": ["https://hf.test/de/test/0.parquet"]}
            )
        return httpx.Response(200, content=b"PARQUETDATA")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = ensure_language_file(GLOBAL, "de", tmp_path, client=client)
    assert path.read_bytes() == b"PARQUETDATA"
    assert path == tmp_path / "global-mmlu" / "de-test.parquet"
    assert not list(tmp_path.rglob("*.part"))


def test_download_failure_raises_dataset_error(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(DatasetError):
        ensure_language_file(GLOBAL, "de", tmp_path, client=client)
    assert not list(tmp_path.rglob("*.parquet*"))


def test_unknown_language_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ensure_language_file(GLOBAL, "xx", tmp_path)


def test_lite_downloads_from_its_own_repo_and_folder(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path.endswith("/parquet/sq"):
            return httpx.Response(
                200, json={"test": ["https://hf.test/sq/test/0.parquet"]}
            )
        return httpx.Response(200, content=b"LITE")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = ensure_language_file(LITE, "sq", tmp_path, client=client)
    assert path == tmp_path / "global-mmlu-lite" / "sq-test.parquet"
    assert "CohereLabs/Global-MMLU-Lite/parquet/sq" in seen[0]


def test_language_must_belong_to_the_dataset(tmp_path):
    with pytest.raises(ValueError):
        ensure_language_file(GLOBAL, "sq", tmp_path)  # Albanian is only in Lite
