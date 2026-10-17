"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Global-MMLU access: per-language Parquet download, local cache, and seeded sampling.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DATASET_ID = "CohereLabs/Global-MMLU"
PARQUET_INDEX_URL = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}"
SPLIT = "test"
DOWNLOAD_TIMEOUT_S = 120.0

# Configs of the dataset (verified 2026-10-17) with their English names.
LANGUAGES: dict[str, str] = {
    "am": "Amharic",
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fil": "Filipino",
    "fr": "French",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "ig": "Igbo",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ky": "Kyrgyz",
    "lt": "Lithuanian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ne": "Nepali",
    "nl": "Dutch",
    "ny": "Chichewa",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sn": "Shona",
    "so": "Somali",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "te": "Telugu",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yo": "Yoruba",
    "zh": "Chinese",
}

COLUMNS = [
    "sample_id",
    "subject",
    "subject_category",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer",
    "cultural_sensitivity_label",
]


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be downloaded or read."""


@dataclass(frozen=True)
class Question:
    """One multiple-choice question of Global-MMLU."""

    sample_id: str
    subject: str
    subject_category: str
    question: str
    options: tuple[str, str, str, str]
    answer: str
    cultural_sensitivity_label: str


def cache_path(cache_dir: Path, language: str) -> Path:
    """Return the local Parquet path of a language's test split."""
    return cache_dir / f"{language}-{SPLIT}.parquet"


def ensure_language_file(
    language: str, cache_dir: Path, client: httpx.Client | None = None
) -> Path:
    """Download the test-split Parquet file of a language unless already cached.

    The file is written to a temporary name and renamed, so an interrupted
    download never leaves a partial file in the cache.

    Args:
        language: dataset config code, for example ``"it"``.
        cache_dir: directory holding cached Parquet files.
        client: optional HTTP client (injected by tests).

    Returns:
        Path: path of the cached Parquet file.

    Raises:
        ValueError: if the language is not supported.
        DatasetError: if the download fails.
    """
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    target = cache_path(cache_dir, language)
    if target.exists():
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    http = client or httpx.Client(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True)
    tmp = target.with_suffix(".parquet.part")
    try:
        index_url = PARQUET_INDEX_URL.format(dataset=DATASET_ID, config=language)
        index = http.get(index_url)
        index.raise_for_status()
        urls = index.json().get(SPLIT) or []
        if len(urls) != 1:
            raise DatasetError(
                f"Expected one Parquet file for {language}/{SPLIT}, found {len(urls)}."
            )
        logger.info("Downloading %s/%s from %s", language, SPLIT, urls[0])
        with http.stream("GET", urls[0]) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        tmp.replace(target)
    except httpx.HTTPError as exc:
        raise DatasetError(f"Download of {language} failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
        if own_client:
            http.close()
    return target


def load_questions(path: Path) -> list[Question]:
    """Read all questions from a cached Parquet file.

    Raises:
        DatasetError: if the file cannot be read or lacks required columns.
    """
    try:
        table = pq.read_table(path, columns=COLUMNS)
    except Exception as exc:  # pyarrow raises several exception types
        raise DatasetError(f"Cannot read {path.name}: {exc}") from exc
    return [
        Question(
            sample_id=row["sample_id"],
            subject=row["subject"],
            subject_category=row["subject_category"],
            question=row["question"],
            options=(
                row["option_a"],
                row["option_b"],
                row["option_c"],
                row["option_d"],
            ),
            answer=row["answer"].strip().upper(),
            cultural_sensitivity_label=row["cultural_sensitivity_label"] or "-",
        )
        for row in table.to_pylist()
    ]


def sample_questions(
    questions: list[Question], num_questions: int, seed: int
) -> list[Question]:
    """Select ``num_questions`` distinct questions, reproducibly for a given seed.

    Raises:
        ValueError: if more questions are requested than available.
    """
    if not 1 <= num_questions <= len(questions):
        raise ValueError(
            f"num_questions must be between 1 and {len(questions)}, got {num_questions}."
        )
    indices = random.Random(seed).sample(range(len(questions)), num_questions)
    return [questions[i] for i in indices]
