"""Environment-backed runtime settings with safe local defaults."""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_DIR / "artifacts" / "model_bundle.joblib"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def model_artifact_path() -> Path:
    """Return the optional environment override or the project-local artifact path."""
    configured_path = os.getenv("PPMI_MODEL_ARTIFACT")
    if configured_path:
        return Path(configured_path)
    return DEFAULT_ARTIFACT_PATH


def api_base_url() -> str:
    """Return the optional API address override or the existing local default."""
    return os.getenv("PPMI_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
