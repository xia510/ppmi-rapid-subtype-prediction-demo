"""Locate optional authorized research fixtures for model-dependent tests."""

import os
from pathlib import Path
from typing import Optional


def authorized_source_root() -> Optional[Path]:
    """Return a validated source directory configured outside the repository."""
    configured = os.getenv("PPMI_TEST_SOURCE_ROOT")
    if not configured:
        return None
    source_root = Path(configured)
    required_files = (
        source_root / "PPMI_4_LASSO_train_raw.csv",
        source_root / "PPMI_4_LASSO_train_1se.csv",
    )
    if not all(path.is_file() for path in required_files):
        return None
    return source_root
