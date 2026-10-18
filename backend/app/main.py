"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: FastAPI application: models, datasets and languages, and the SSE evaluation endpoint.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .dataset import (
    DATASETS,
    DEFAULT_DATASET,
    LANGUAGE_NAMES,
    DatasetError,
    DatasetInfo,
    Question,
    cache_path,
    ensure_language_file,
    load_questions,
    sample_questions,
)
from .evaluator import RunConfig, run_evaluation
from .llm_client import REASONING_EFFORTS, ChatModel, OciChatModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Characters allowed in run IDs, which are also result file names.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings."""
    return load_settings()


@lru_cache
def get_chat_model() -> ChatModel:
    """Return the process-wide OCI chat model (created on first use)."""
    return OciChatModel(get_settings())


class QuestionStore:
    """In-memory cache of loaded (dataset, language) pairs, one download at a time each."""

    def __init__(self):
        self._questions: dict[tuple[str, str], list[Question]] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def is_cached(
        self, dataset: DatasetInfo, language: str, settings: Settings
    ) -> bool:
        """Tell whether the language is already available without a download."""
        return (dataset.key, language) in self._questions or cache_path(
            settings.data_cache_dir, dataset, language
        ).exists()

    async def get(
        self, dataset: DatasetInfo, language: str, settings: Settings
    ) -> list[Question]:
        """Return all questions of a language, downloading the file if needed."""
        key = (dataset.key, language)
        if key in self._questions:
            return self._questions[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self._questions:
                path = await asyncio.to_thread(
                    ensure_language_file, dataset, language, settings.data_cache_dir
                )
                self._questions[key] = await asyncio.to_thread(load_questions, path)
        return self._questions[key]


@lru_cache
def get_store() -> QuestionStore:
    """Return the process-wide question store."""
    return QuestionStore()


class EvaluationRequest(BaseModel):
    """Body of ``POST /api/evaluations``."""

    model_id: str
    dataset: str = DEFAULT_DATASET
    language: str
    num_questions: int = Field(ge=1)
    # Required, so every run can be repeated; the UI pre-fills DEFAULT_SEED.
    seed: int = Field(ge=0)
    # None: the parameter is not sent and the model uses its own default.
    reasoning_effort: Literal["NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH"] | None = None


app = FastAPI(title="Global-MMLU multilingual evaluation", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/api/models")
def list_models(settings: Settings = Depends(get_settings)) -> dict:
    """Return the configured model IDs and the default one."""
    return {
        "models": settings.models,
        "default": settings.default_model,
        "region": settings.oci_region,
        "reasoning_efforts": list(REASONING_EFFORTS),
        "default_seed": settings.default_seed,
    }


def _languages(dataset: DatasetInfo) -> list[dict]:
    """Return the languages of a dataset as code and English name."""
    return [
        {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
        for code in dataset.languages
    ]


@app.get("/api/datasets")
def list_datasets() -> dict:
    """Return the supported datasets with their languages and test-split size."""
    return {
        "default": DEFAULT_DATASET,
        "datasets": [
            {
                "key": d.key,
                "name": d.name,
                "hf_id": d.hf_id,
                "test_rows": d.test_rows,
                "languages": _languages(d),
            }
            for d in DATASETS.values()
        ],
    }


@app.get("/api/languages")
def list_languages(dataset: str = Query(DEFAULT_DATASET)) -> dict:
    """Return the language codes of a dataset with English names.

    Raises:
        HTTPException: 422 for an unknown dataset.
    """
    if dataset not in DATASETS:
        raise HTTPException(422, f"Unknown dataset: {dataset}")
    return {"languages": _languages(DATASETS[dataset])}


def _sse(event: dict) -> str:
    """Encode one event as a Server-Sent Events message."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _event_stream(
    request: EvaluationRequest,
    settings: Settings,
    model: ChatModel,
    store: QuestionStore,
) -> AsyncIterator[str]:
    """Prepare the sample and relay evaluator events as SSE messages."""
    dataset = DATASETS[request.dataset]
    try:
        if not store.is_cached(dataset, request.language, settings):
            yield _sse(
                {
                    "type": "status",
                    "message": f"Downloading {dataset.name} '{request.language}' test split",
                }
            )
        questions = await store.get(dataset, request.language, settings)
        seed = request.seed
        sample = sample_questions(questions, request.num_questions, seed)
        run_id = (
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{dataset.key}-"
            f"{request.language}-{_SAFE_NAME_RE.sub('_', request.model_id)}-{seed}"
        )
        config = RunConfig(
            run_id=run_id,
            model_id=request.model_id,
            model_target=settings.model_targets.get(request.model_id, request.model_id),
            reasoning_effort=request.reasoning_effort,
            dataset=dataset.key,
            language=request.language,
            seed=seed,
            max_concurrency=settings.max_concurrency,
            results_dir=settings.results_dir,
        )
        logger.info("Run %s: %d questions", run_id, len(sample))
        async for event in run_evaluation(model, sample, config):
            yield _sse(event)
    except (DatasetError, ValueError) as exc:
        logger.error("Evaluation failed: %s", exc)
        yield _sse({"type": "error", "message": str(exc)})


@app.post("/api/evaluations")
def create_evaluation(
    request: EvaluationRequest,
    settings: Settings = Depends(get_settings),
    store: QuestionStore = Depends(get_store),
) -> StreamingResponse:
    """Start an evaluation and stream its progress as Server-Sent Events.

    Raises:
        HTTPException: 422 for an unknown model, dataset, or language, or too
            many questions.
    """
    if request.model_id not in settings.models:
        raise HTTPException(422, f"Unknown model: {request.model_id}")
    dataset = DATASETS.get(request.dataset)
    if dataset is None:
        raise HTTPException(422, f"Unknown dataset: {request.dataset}")
    if request.language not in dataset.languages:
        raise HTTPException(
            422, f"{dataset.name} does not include language: {request.language}"
        )
    limit = min(settings.max_questions, dataset.test_rows)
    if request.num_questions > limit:
        raise HTTPException(
            422, f"num_questions must be at most {limit} for {dataset.name}"
        )
    try:
        model = get_chat_model()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # OCI config problems surface here; report them without credential details.
        raise HTTPException(
            500, f"OCI client setup failed: {type(exc).__name__}"
        ) from exc
    return StreamingResponse(
        _event_stream(request, settings, model, store),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
