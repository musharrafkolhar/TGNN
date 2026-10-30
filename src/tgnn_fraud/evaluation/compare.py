"""Compare measured model outputs only after verifying common evaluation events."""

import argparse
import json
from pathlib import Path

import pandas as pd

from tgnn_fraud.evaluation.metrics import validate_comparison
from tgnn_fraud.evaluation.run_eval import run_evaluation
from tgnn_fraud.utils.config import write_json


def compare_runs(runs: list[Path], output: Path) -> dict:
    """Check dataset identity and held-out joins before saving actual metrics."""
    if len(runs) < 2:
        raise ValueError("Comparison requires at least two model runs")
    frames = [pd.read_parquet(run / "predictions.parquet") for run in runs]
    manifests = [
        json.loads((run / "data_manifest.json").read_text(encoding="utf-8")) for run in runs
    ]
    for frame, manifest in zip(frames[1:], manifests[1:], strict=True):
        if manifest["edges_sha256"] != manifests[0]["edges_sha256"]:
            raise ValueError("Comparison requires identical canonical dataset hashes")
        validate_comparison(frames[0], frame)
    results = [run_evaluation(run) for run in runs]
    result = {
        "source_kind": manifests[0]["source_kind"],
        "edges_sha256": manifests[0]["edges_sha256"],
        "held_out_sets_verified_identical": True,
        "interpretation": "Measured scores on this subset only; no temporal superiority assumed.",
        "models": results,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/comparison.json"))
    args = parser.parse_args()
    compare_runs(args.runs, args.output)


if __name__ == "__main__":
    main()
