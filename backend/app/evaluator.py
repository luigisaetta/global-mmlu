"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Prompt building, answer grading, and the streaming evaluation loop.
"""

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .dataset import Question
from .llm_client import ChatModel, InvalidRequestError

logger = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
INSTRUCTION = (
    "The following is a multiple choice question. Reply with only the letter "
    "(A, B, C, or D) of the correct answer."
)
_LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def build_prompt(question: Question) -> str:
    """Return the prompt for one question (English instruction, original-language content)."""
    options = "\n".join(
        f"{letter}. {text}" for letter, text in zip(LETTERS, question.options)
    )
    return f"{INSTRUCTION}\n\n{question.question}\n\n{options}"


def extract_letter(reply: str) -> str | None:
    """Return the first standalone letter A-D in the reply, or None if there is none."""
    match = _LETTER_RE.search(reply.strip())
    return match.group(1) if match else None


@dataclass
class RunConfig:  # pylint: disable=too-many-instance-attributes
    """Inputs of one evaluation run.

    ``model_id`` is the label shown to the user; ``model_target`` is what the
    chat model is called with (model ID or endpoint OCID), defaulting to the label.
    ``reasoning_effort`` None means the parameter is not sent (model default).
    """

    run_id: str
    model_id: str
    language: str
    seed: int
    max_concurrency: int
    results_dir: Path
    model_target: str = ""
    reasoning_effort: str | None = None

    @property
    def target(self) -> str:
        """Return the invocation target of the run."""
        return self.model_target or self.model_id


@dataclass
class _Tally:
    """Running counters for a group of results."""

    total: int = 0
    correct: int = 0

    def add(self, correct: bool) -> None:
        """Count one result."""
        self.total += 1
        self.correct += int(correct)

    def as_dict(self) -> dict:
        """Return counters and accuracy as a JSON-friendly dict."""
        accuracy = self.correct / self.total if self.total else 0.0
        return {"total": self.total, "correct": self.correct, "accuracy": accuracy}


@dataclass
class _RunState:
    """Aggregated state of a run while results arrive."""

    overall: _Tally = field(default_factory=_Tally)
    by_category: dict = field(default_factory=lambda: defaultdict(_Tally))
    by_sensitivity: dict = field(default_factory=lambda: defaultdict(_Tally))
    invalid: int = 0
    errors: int = 0
    latency_sum: float = 0.0
    results: list = field(default_factory=list)


def _grade(
    index: int,
    question: Question,
    reply: str | None,
    error: str | None,
    latency_s: float,
) -> dict:
    """Build the result record of one question."""
    predicted = extract_letter(reply) if reply is not None else None
    return {
        "index": index,
        "sample_id": question.sample_id,
        "subject": question.subject,
        "subject_category": question.subject_category,
        "cultural_sensitivity_label": question.cultural_sensitivity_label,
        "expected": question.answer,
        "predicted": predicted,
        "correct": predicted == question.answer,
        "invalid": error is None and predicted is None,
        "latency_s": round(latency_s, 3),
        "error": error,
        "reply": (reply or "")[:200],
    }


async def _ask(
    model: ChatModel,
    config: RunConfig,
    index: int,
    question: Question,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run one call in a worker thread, bounded by the semaphore, and grade it.

    Raises:
        InvalidRequestError: if the service rejects the request itself.
    """
    async with semaphore:
        start = time.perf_counter()
        try:
            reply = await asyncio.to_thread(
                model.complete,
                config.target,
                build_prompt(question),
                config.reasoning_effort,
            )
            error = None
        except InvalidRequestError:
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # A failed call is recorded and counted as wrong; the run continues.
            logger.warning("Call for %s failed: %s", question.sample_id, exc)
            reply, error = None, f"{type(exc).__name__}: {str(exc)[:300]}"
        return _grade(index, question, reply, error, time.perf_counter() - start)


def _summary(config: RunConfig, state: _RunState, total: int) -> dict:
    """Build the final summary of a run."""
    completed = state.overall.total
    return {
        "run_id": config.run_id,
        "model_id": config.model_id,
        "model_target": config.target,
        "reasoning_effort": config.reasoning_effort,
        "language": config.language,
        "seed": config.seed,
        "requested": total,
        "completed": completed,
        **state.overall.as_dict(),
        "by_subject_category": {
            k: v.as_dict() for k, v in sorted(state.by_category.items())
        },
        "by_cultural_sensitivity": {
            k: v.as_dict() for k, v in sorted(state.by_sensitivity.items())
        },
        "invalid": state.invalid,
        "errors": state.errors,
        "mean_latency_s": round(state.latency_sum / completed, 3) if completed else 0.0,
    }


def _save(config: RunConfig, summary: dict, results: list) -> Path:
    """Write the run summary and per-question results to a JSON file."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.results_dir / f"{config.run_id}.json"
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": sorted(results, key=lambda r: r["index"]),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def run_evaluation(
    model: ChatModel, questions: list[Question], config: RunConfig
) -> AsyncIterator[dict]:
    """Evaluate the questions and yield progress events.

    Yields a ``start`` event, one ``result`` event per question in completion
    order (with running accuracy), and a final ``done`` event. If the service
    rejects the request itself (HTTP 400), it yields an ``error`` event and
    stops. If the consumer stops iterating (client disconnect), pending calls
    that have not started are cancelled. Partial results are saved in both
    cases.

    Args:
        model: chat model used for the calls.
        questions: sampled questions, in sample order.
        config: run inputs.

    Yields:
        dict: events with a ``type`` field.
    """
    total = len(questions)
    yield {
        "type": "start",
        "run_id": config.run_id,
        "model_id": config.model_id,
        "reasoning_effort": config.reasoning_effort,
        "language": config.language,
        "seed": config.seed,
        "total": total,
    }

    semaphore = asyncio.Semaphore(config.max_concurrency)
    tasks = [
        asyncio.create_task(_ask(model, config, i, q, semaphore))
        for i, q in enumerate(questions)
    ]
    state = _RunState()
    try:
        for next_done in asyncio.as_completed(tasks):
            try:
                result = await next_done
            except InvalidRequestError as exc:
                logger.error("Run %s rejected by the service: %s", config.run_id, exc)
                hint = (
                    " The model may not accept this reasoning effort;"
                    " try 'Model default'."
                    if config.reasoning_effort
                    else ""
                )
                yield {"type": "error", "message": f"Request rejected: {exc}.{hint}"}
                return
            state.results.append(result)
            state.overall.add(result["correct"])
            state.by_category[result["subject_category"]].add(result["correct"])
            state.by_sensitivity[result["cultural_sensitivity_label"]].add(
                result["correct"]
            )
            state.invalid += int(result["invalid"])
            state.errors += int(result["error"] is not None)
            state.latency_sum += result["latency_s"]
            yield {
                "type": "result",
                **result,
                "completed": state.overall.total,
                "correct_count": state.overall.correct,
                "accuracy": state.overall.as_dict()["accuracy"],
            }
    finally:
        for task in tasks:
            task.cancel()
        if state.overall.total < total:
            # Stopped early (client disconnect or rejected request): keep the
            # partial results.
            summary = _summary(config, state, total) | {"stopped": True}
            path = _save(config, summary, state.results)
            logger.info(
                "Run %s stopped after %d/%d questions, saved to %s",
                config.run_id,
                state.overall.total,
                total,
                path.name,
            )

    summary = _summary(config, state, total) | {"stopped": False}
    path = _save(config, summary, state.results)
    logger.info(
        "Run %s done: accuracy %.3f (%d/%d)",
        config.run_id,
        summary["accuracy"],
        summary["correct"],
        total,
    )
    yield {"type": "done", **summary, "result_file": str(path)}
