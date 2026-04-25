from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - fallback for older/newer sklearn variants
    InconsistentVersionWarning = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sklearn_pickle(
    path: str | Path,
    *,
    expected_hash: str = "",
    enforce_version_match: bool = True,
) -> Any:
    actual_hash = sha256_file(path)
    if expected_hash and actual_hash.lower() != expected_hash.lower():
        raise ValueError(f"Integrity check failed for {path}")

    with warnings.catch_warnings():
        if enforce_version_match and InconsistentVersionWarning is not None:
            warnings.filterwarnings("error", category=InconsistentVersionWarning)
        with open(path, "rb") as file_obj:
            return pickle.load(file_obj)


def load_sklearn_joblib(
    path: str | Path,
    *,
    enforce_version_match: bool = True,
) -> Any:
    with warnings.catch_warnings():
        if enforce_version_match and InconsistentVersionWarning is not None:
            warnings.filterwarnings("error", category=InconsistentVersionWarning)
        return joblib.load(path)
