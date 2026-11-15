"""Shared helpers: configuration, logging, paths, hashing and JSON IO."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    with open(config_path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def path(key: str, *parts: str) -> Path:
    """Resolve a configured directory (e.g. ``path("raw", "users.csv")``) and make sure it exists."""
    base = PROJECT_ROOT / load_config()["paths"][key]
    result = base.joinpath(*parts)
    (result.parent if parts else result).mkdir(parents=True, exist_ok=True)
    return result


def setup_mlflow():
    """Point MLflow at the project's SQLite tracking store and create the experiment if needed."""
    import mlflow

    cfg = load_config()
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / cfg['paths']['mlflow_db']}")
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = PROJECT_ROOT / cfg["paths"]["mlflow_artifacts"]
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)
    return mlflow


def set_seed(seed: int | None = None) -> int:
    seed = load_config()["project"]["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(path("logs", "pipeline.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def file_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(obj: Any, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def read_json(file_path: Path) -> Any:
    with open(file_path) as fh:
        return json.load(fh)
