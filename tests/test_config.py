"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Tests of MODELS parsing and of the serving mode chosen for a target.
"""

# pylint: disable=missing-function-docstring

import pytest
from oci.generative_ai_inference import models as genai

from app.config import parse_models
from app.llm_client import serving_mode

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
