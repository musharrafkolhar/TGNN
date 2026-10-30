"""Configuration, JSON and reproducible run creation."""

import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import yaml


def read_config(path: str | Path) -> dict:
    """Read YAML paths relative to the project working directory."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    return config


def write_json(path: Path, value: dict | list) -> None:
    """Write strict JSON, rejecting NaN and infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path: Path) -> str:
    """Return streaming SHA256 without loading a large file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run(config: dict, model: str) -> Path:
    """Create a collision-safe run folder and record environment provenance."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(config["runs_dir"]) / f"{stamp}_{model}_{uuid4().hex[:6]}"
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (path / "seed.txt").write_text(str(config["seed"]), encoding="utf-8")
    packages = {}
    for name in ["torch", "torch-geometric", "numpy", "pandas", "xgboost", "scikit-learn"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    write_json(path / "environment.json", {"python": platform.python_version(), **packages})
    return path


def configure_logging() -> None:
    """Enable concise command-line progress logs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
