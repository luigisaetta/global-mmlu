"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Tests of the FastAPI endpoints with a fake model and local data.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import main
from conftest import FakeChatModel


@pytest.fixture
def client(parquet_file, tmp_path, monkeypatch):
    settings = replace(
        main.get_settings(),
        models=["fake.model", "dedicated"],
        default_model="fake.model",
        model_targets={
            "fake.model": "fake.model",
            "dedicated": "ocid1.generativeaiendpoint.oc1.test.example",
        },
        max_questions=20,
        data_cache_dir=parquet_file,
        results_dir=tmp_path / "results",
    )
    fake = FakeChatModel()
    monkeypatch.setattr(main, "get_chat_model", lambda: fake)
    main.app.dependency_overrides[main.get_settings] = lambda: settings
    main.app.dependency_overrides[main.get_store] = main.QuestionStore
    yield TestClient(main.app), fake
    main.app.dependency_overrides.clear()


def _events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_models_and_languages(client):
    http, _ = client
    assert http.get("/api/models").json()["models"] == ["fake.model", "dedicated"]
    codes = [lang["code"] for lang in http.get("/api/languages").json()["languages"]]
    assert len(codes) == 42 and "it" in codes


def test_evaluation_streams_events(client):
    http, fake = client
    body = {"model_id": "fake.model", "language": "it", "num_questions": 5, "seed": 3}
    response = http.post("/api/evaluations", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)
    assert [e["type"] for e in events] == ["start"] + ["result"] * 5 + ["done"]
    assert events[0]["seed"] == 3 and fake.calls == 5


@pytest.mark.parametrize(
    "body",
    [
        {"model_id": "other", "language": "it", "num_questions": 5},
        {"model_id": "fake.model", "language": "xx", "num_questions": 5},
        {"model_id": "fake.model", "language": "it", "num_questions": 21},
        {"model_id": "fake.model", "language": "it", "num_questions": 0},
        {
            "model_id": "fake.model",
            "language": "it",
            "num_questions": 5,
            "reasoning_effort": "EXTREME",
        },
    ],
)
def test_invalid_requests_are_rejected_without_calls(client, body):
    http, fake = client
    assert http.post("/api/evaluations", json=body).status_code == 422
    assert fake.calls == 0


def test_dedicated_label_calls_the_endpoint_ocid(client):
    http, fake = client
    body = {"model_id": "dedicated", "language": "it", "num_questions": 2, "seed": 1}
    events = _events(http.post("/api/evaluations", json=body))
    assert events[-1]["model_id"] == "dedicated"
    assert events[-1]["model_target"] == "ocid1.generativeaiendpoint.oc1.test.example"
    assert set(fake.model_ids) == {"ocid1.generativeaiendpoint.oc1.test.example"}


def test_reasoning_effort_reaches_the_model(client):
    http, fake = client
    body = {
        "model_id": "fake.model",
        "language": "it",
        "num_questions": 3,
        "reasoning_effort": "NONE",
    }
    events = _events(http.post("/api/evaluations", json=body))
    assert events[0]["reasoning_effort"] == "NONE"
    assert events[-1]["reasoning_effort"] == "NONE"
    assert fake.efforts == ["NONE"] * 3


def test_default_reasoning_effort_is_not_sent(client):
    http, fake = client
    body = {"model_id": "fake.model", "language": "it", "num_questions": 2}
    http.post("/api/evaluations", json=body)
    assert fake.efforts == [None, None]


def test_rejected_request_stops_the_run(client):
    http, fake = client
    fake.reject = True
    body = {
        "model_id": "fake.model",
        "language": "it",
        "num_questions": 20,
        "reasoning_effort": "LOW",
    }
    events = _events(http.post("/api/evaluations", json=body))
    assert [e["type"] for e in events] == ["start", "error"]
    assert "Model default" in events[-1]["message"]
    assert fake.calls < 20


def test_datasets_list_languages_and_sizes(client):
    http, _ = client
    payload = http.get("/api/datasets").json()
    by_key = {d["key"]: d for d in payload["datasets"]}
    assert payload["default"] == "global-mmlu"
    assert by_key["global-mmlu"]["test_rows"] == 14327
    lite_codes = [lang["code"] for lang in by_key["global-mmlu-lite"]["languages"]]
    assert len(lite_codes) == 23 and "sq" in lite_codes
    lite = http.get("/api/languages", params={"dataset": "global-mmlu-lite"}).json()
    assert {"code": "sq", "name": "Albanian"} in lite["languages"]
    assert http.get("/api/languages", params={"dataset": "other"}).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"model_id": "fake.model", "dataset": "other", "language": "it"},
        {"model_id": "fake.model", "dataset": "global-mmlu", "language": "sq"},
        {"model_id": "fake.model", "dataset": "global-mmlu-lite", "language": "am"},
    ],
)
def test_language_must_match_dataset(client, body):
    http, fake = client
    response = http.post("/api/evaluations", json=body | {"num_questions": 5})
    assert response.status_code == 422
    assert fake.calls == 0


def test_lite_caps_questions_at_its_size(client):
    http, _ = client
    settings = main.app.dependency_overrides[main.get_settings]()
    main.app.dependency_overrides[main.get_settings] = lambda: replace(
        settings, max_questions=1000
    )
    body = {
        "model_id": "fake.model",
        "dataset": "global-mmlu-lite",
        "language": "sq",
        "num_questions": 401,
    }
    response = http.post("/api/evaluations", json=body)
    assert response.status_code == 422
    assert "400" in response.json()["detail"]
