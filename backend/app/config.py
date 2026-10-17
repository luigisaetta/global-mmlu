"""
Author: L. Saetta
Date last modified: 2026-10-17
License: MIT
Description: Backend settings loaded from environment variables and the root .env file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated value into a list of non-empty, stripped items."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_path(value: str) -> Path:
    """Expand ``~`` and resolve relative paths against the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:  # pylint: disable=too-many-instance-attributes
    """Runtime settings of the evaluation backend."""

    oci_config_file: str
    oci_profile: str
    oci_region: str
    oci_compartment_id: str | None
    models: list[str]
    default_model: str
    max_questions: int
    max_concurrency: int
    max_completion_tokens: int
    data_cache_dir: Path
    results_dir: Path
    cors_origins: list[str] = field(default_factory=list)


def load_settings() -> Settings:
    """Build the settings from the environment, reading the root .env if present.

    Returns:
        Settings: validated settings.

    Raises:
        ValueError: if the model list is empty or the default model is not in it.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    models = _split_csv(os.getenv("MODELS", "openai.gpt-5.6-luna"))
    default_model = os.getenv("DEFAULT_MODEL", "").strip() or (
        models[0] if models else ""
    )
    if not models:
        raise ValueError("MODELS must list at least one model ID.")
    if default_model not in models:
        raise ValueError(f"DEFAULT_MODEL '{default_model}' is not listed in MODELS.")
    return Settings(
        oci_config_file=os.path.expanduser(
            os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
        ),
        oci_profile=os.getenv("OCI_PROFILE", "DEFAULT"),
        oci_region=os.getenv("OCI_REGION", "eu-frankfurt-1"),
        oci_compartment_id=os.getenv("OCI_COMPARTMENT_ID", "").strip() or None,
        models=models,
        default_model=default_model,
        max_questions=int(os.getenv("MAX_QUESTIONS", "500")),
        max_concurrency=max(1, int(os.getenv("MAX_CONCURRENCY", "4"))),
        max_completion_tokens=int(os.getenv("MAX_COMPLETION_TOKENS", "2000")),
        data_cache_dir=_resolve_path(os.getenv("DATA_CACHE_DIR", "data/cache")),
        results_dir=_resolve_path(os.getenv("RESULTS_DIR", "results")),
        cors_origins=_split_csv(
            os.getenv("CORS_ORIGINS", "").strip()
            or "http://localhost:3000,http://127.0.0.1:3000"
        ),
    )
