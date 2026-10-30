"""Copy small saved demo outputs, without fabricated data or full training inputs."""

import argparse
import json
import shutil
from pathlib import Path

from tgnn_fraud.utils.config import read_config


def build_bundle(run: Path, config: dict) -> Path:
    """Bundle real metrics, alerts and provenance; checkpoint is not needed by the UI."""
    output = Path("data/demo") / run.name
    output.mkdir(parents=True, exist_ok=True)
    for name in [
        "alerts.db",
        "metrics.json",
        "alert_export.json",
        "data_manifest.json",
        "config.yaml",
        "environment.json",
        "seed.txt",
        "predictions.parquet",
        "pr_curve.html",
    ]:
        shutil.copy2(run / name, output / name)
    report = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
    if report["source_kind"] == "test_fixture":
        (output / "FIXTURE_ONLY.txt").write_text(
            "This bundle tests the pipeline using planted synthetic fixtures. Not an IBM result.",
            encoding="utf-8",
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build_bundle(args.run, read_config(args.config))


if __name__ == "__main__":
    main()
