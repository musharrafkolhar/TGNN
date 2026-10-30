"""Explicit staged orchestration; errors stop execution without silent fallback."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tgnn_fraud.data.preprocess import preprocess  # noqa: E402
from tgnn_fraud.data.synthetic import make_fixture  # noqa: E402
from tgnn_fraud.evaluation.compare import compare_runs  # noqa: E402
from tgnn_fraud.evaluation.run_eval import run_evaluation  # noqa: E402
from tgnn_fraud.explain.run_explain import export_alerts  # noqa: E402
from tgnn_fraud.serving.build_demo_bundle import build_bundle  # noqa: E402
from tgnn_fraud.utils.config import configure_logging, read_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prototype.yaml"))
    parser.add_argument(
        "--fixture", action="store_true", help="Generate planted test fixture, not IBM data"
    )
    parser.add_argument("--model", choices=["xgboost", "tgn", "both"], default="both")
    parser.add_argument("--input", type=Path, help="Override the configured transaction CSV")
    args = parser.parse_args()
    configure_logging()
    config = read_config(args.config)
    if args.input:
        config["data"]["input"] = str(args.input.resolve())
    if args.fixture:
        if config["data"]["source_kind"] != "test_fixture":
            raise ValueError(
                "--fixture requires source_kind: test_fixture; do not overwrite IBM data"
            )
        source = Path(config["data"]["input"])
        source.parent.mkdir(parents=True, exist_ok=True)
        make_fixture(seed=config["seed"]).to_csv(source, index=False)
    preprocess(config)
    models = ["xgboost", "tgn"] if args.model == "both" else [args.model]
    runs = []
    for model in models:
        if model == "xgboost":
            from tgnn_fraud.training.train_baselines import train
        else:
            from tgnn_fraud.training.train_tgn import train
        run = train(config)
        runs.append(run)
        run_evaluation(run)
        export_alerts(run, config)
        build_bundle(run, config)
    if len(runs) > 1:
        compare_runs(runs, Path(config["runs_dir"]) / "comparison.json")


if __name__ == "__main__":
    main()
