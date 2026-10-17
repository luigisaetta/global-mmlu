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
DEDICATED_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint."
# Values accepted by GenericChatRequest.reasoning_effort (OCI SDK 2.188.0).
REASONING_EFFORTS = ("NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH")


class InvalidRequestError(RuntimeError):
    """The service rejected the request itself (HTTP 400).

    The same request fails for every question, for example when a model does
    not accept ``reasoning_effort``, so the evaluator stops the run.
    """


def serving_mode(target: str) -> genai.ServingMode:
    """Return the serving mode for a target.

    Args:
        target: an on-demand model ID, or the OCID of a dedicated AI cluster
            endpoint.

    Returns:
        genai.ServingMode: ``DedicatedServingMode`` for endpoint OCIDs,
        ``OnDemandServingMode`` otherwise.
    """
    if target.startswith(DEDICATED_ENDPOINT_PREFIX):
        return genai.DedicatedServingMode(endpoint_id=target)
    return genai.OnDemandServingMode(model_id=target)


class ChatModel(Protocol):
    """Minimal interface used by the evaluator, so tests can use a fake."""

    def complete(
        self, model_id: str, prompt: str, reasoning_effort: str | None = None
    ) -> str:
        """Send one user prompt to a model ID or endpoint OCID; return the reply.

        ``reasoning_effort`` is sent only when not None.
        """


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

    def build_request(
        self, model_id: str, prompt: str, reasoning_effort: str | None = None
    ) -> genai.GenericChatRequest:
        """Build the chat request.

        On-demand OpenAI models take ``max_completion_tokens``; other models,
        including dedicated endpoints, take ``max_tokens`` and temperature 0.
        ``reasoning_effort`` is added only when given: some models reject it
        (verified 2026-10-17: ``openai.gpt-4.1`` for any value, a dedicated
        gpt-oss endpoint for ``NONE``), others ignore it (``meta.llama-3.3``).
        """
        message = genai.UserMessage(content=[genai.TextContent(text=prompt)])
        if model_id.startswith("openai."):
            options = {"max_completion_tokens": self._max_completion_tokens}
        else:
            options = {"max_tokens": self._max_completion_tokens, "temperature": 0.0}
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort
        return genai.GenericChatRequest(
            api_format=genai.BaseChatRequest.API_FORMAT_GENERIC,
            messages=[message],
            **options,
        )

    def complete(
        self, model_id: str, prompt: str, reasoning_effort: str | None = None
    ) -> str:
        """Send one prompt to a model ID or endpoint OCID and return the reply text.

        Raises:
            InvalidRequestError: if the service rejects the request (HTTP 400).
            oci.exceptions.ServiceError: on other service errors left after retries.
        """
        details = genai.ChatDetails(
            compartment_id=self._compartment_id,
            serving_mode=serving_mode(model_id),
            chat_request=self.build_request(model_id, prompt, reasoning_effort),
        )
        try:
            response = self._client().chat(details)
        except oci.exceptions.ServiceError as exc:
            if exc.status == 400:
                raise InvalidRequestError(_service_message(exc)) from exc
            raise
        choices = response.data.chat_response.choices or []
        if not choices or not choices[0].message or not choices[0].message.content:
            return ""
        return "".join(
            part.text
            for part in choices[0].message.content
            if getattr(part, "text", None)
        )


def _service_message(exc: oci.exceptions.ServiceError) -> str:
    """Return a short, single-line message of a service error."""
    message = " ".join(str(exc.message or exc.code).split())
    return f"HTTP {exc.status}: {message[:300]}"
