"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Chat invocation of OCI Generative AI models with user principal authentication.
"""

import threading
from typing import Protocol

import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference import models as genai

from .config import Settings

INFERENCE_ENDPOINT = "https://inference.generativeai.{region}.oci.oraclecloud.com"


class ChatModel(Protocol):
    """Minimal interface used by the evaluator, so tests can use a fake."""

    def complete(self, model_id: str, prompt: str) -> str:
        """Send one user prompt and return the text of the reply."""


class OciChatModel:
    """Calls OCI Generative AI chat with the GENERIC API format.

    Authentication uses the API key of an OCI config profile (user principal).
    One SDK client is created per worker thread, because the evaluator runs
    calls in a thread pool.
    """

    def __init__(self, settings: Settings):
        """Load the OCI config and resolve the compartment.

        Raises:
            oci.exceptions.ConfigFileNotFound: if the config file is missing.
            oci.exceptions.ProfileNotFound: if the profile does not exist.
            oci.exceptions.InvalidConfig: if the profile is incomplete.
        """
        config = oci.config.from_file(settings.oci_config_file, settings.oci_profile)
        config["region"] = settings.oci_region
        oci.config.validate_config(config)
        self._config = config
        self._endpoint = INFERENCE_ENDPOINT.format(region=settings.oci_region)
        self._compartment_id = settings.oci_compartment_id or config["tenancy"]
        self._max_completion_tokens = settings.max_completion_tokens
        self._local = threading.local()

    def _client(self) -> GenerativeAiInferenceClient:
        """Return the SDK client of the current thread, creating it on first use."""
        client = getattr(self._local, "client", None)
        if client is None:
            client = GenerativeAiInferenceClient(
                self._config,
                service_endpoint=self._endpoint,
                retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
                timeout=(10, 120),
            )
            self._local.client = client
        return client

    def _build_request(self, model_id: str, prompt: str) -> genai.GenericChatRequest:
        """Build the chat request; OpenAI models take completion-token and reasoning options."""
        message = genai.UserMessage(content=[genai.TextContent(text=prompt)])
        if model_id.startswith("openai."):
            return genai.GenericChatRequest(
                api_format=genai.BaseChatRequest.API_FORMAT_GENERIC,
                messages=[message],
                max_completion_tokens=self._max_completion_tokens,
                reasoning_effort="LOW",
            )
        return genai.GenericChatRequest(
            api_format=genai.BaseChatRequest.API_FORMAT_GENERIC,
            messages=[message],
            max_tokens=self._max_completion_tokens,
            temperature=0.0,
        )

    def complete(self, model_id: str, prompt: str) -> str:
        """Send one prompt to an on-demand model and return the reply text.

        Raises:
            oci.exceptions.ServiceError: on service errors left after retries.
        """
        details = genai.ChatDetails(
            compartment_id=self._compartment_id,
            serving_mode=genai.OnDemandServingMode(model_id=model_id),
            chat_request=self._build_request(model_id, prompt),
        )
        response = self._client().chat(details)
        choices = response.data.chat_response.choices or []
        if not choices or not choices[0].message or not choices[0].message.content:
            return ""
        return "".join(
            part.text
            for part in choices[0].message.content
            if getattr(part, "text", None)
        )
