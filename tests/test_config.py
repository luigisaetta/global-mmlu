"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Tests of MODELS parsing and of the serving mode chosen for a target.
"""

# pylint: disable=missing-function-docstring

import pytest
from oci.generative_ai_inference import models as genai

from app.config import load_settings, parse_models
from app.llm_client import OciChatModel, serving_mode

ENDPOINT = "ocid1.generativeaiendpoint.oc1.eu-frankfurt-1.example"


def test_plain_ids_use_the_id_as_label():
    assert parse_models("openai.gpt-5.6-luna, openai.gpt-5.5") == {
        "openai.gpt-5.6-luna": "openai.gpt-5.6-luna",
        "openai.gpt-5.5": "openai.gpt-5.5",
    }


def test_labelled_entries_map_to_their_target():
    parsed = parse_models(f"openai.gpt-5.6-luna,gpt-oss-dac={ENDPOINT}")
    assert parsed["gpt-oss-dac"] == ENDPOINT
    assert list(parsed) == ["openai.gpt-5.6-luna", "gpt-oss-dac"]


@pytest.mark.parametrize("value", ["=ocid1.x", "label=", "a,a", "a=x,a=y"])
def test_invalid_entries_are_rejected(value):
    with pytest.raises(ValueError):
        parse_models(value)


def test_serving_mode_for_on_demand_and_dedicated():
    on_demand = serving_mode("openai.gpt-5.6-luna")
    dedicated = serving_mode(ENDPOINT)
    assert isinstance(on_demand, genai.OnDemandServingMode)
    assert on_demand.model_id == "openai.gpt-5.6-luna"
    assert isinstance(dedicated, genai.DedicatedServingMode)
    assert dedicated.endpoint_id == ENDPOINT


def _client_with_budget(tokens: int) -> OciChatModel:
    """Return an OciChatModel without OCI config, enough for build_request."""
    client = object.__new__(OciChatModel)
    client._max_completion_tokens = tokens  # pylint: disable=protected-access
    return client


def test_reasoning_effort_is_sent_only_when_given():
    client = _client_with_budget(500)
    default = client.build_request("meta.llama-3.3-70b-instruct", "q")
    explicit = client.build_request("meta.llama-3.3-70b-instruct", "q", "NONE")
    assert default.reasoning_effort is None
    assert default.max_tokens == 500 and default.temperature == 0.0
    assert explicit.reasoning_effort == "NONE"


def test_openai_models_use_max_completion_tokens_without_forced_effort():
    request = _client_with_budget(500).build_request("openai.gpt-4.1", "q")
    assert request.max_completion_tokens == 500
    assert request.max_tokens is None
    assert request.reasoning_effort is None


def test_default_seed_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DEFAULT_SEED", "7")
    assert load_settings().default_seed == 7


@pytest.mark.parametrize("value", ["-1", "abc", "4.2"])
def test_invalid_default_seed_is_rejected(monkeypatch, value):
    monkeypatch.setenv("DEFAULT_SEED", value)
    with pytest.raises(ValueError):
        load_settings()
