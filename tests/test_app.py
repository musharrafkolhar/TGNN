"""Execute Streamlit UI views using actual saved baseline outputs."""

from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from tgnn_fraud.data.preprocess import preprocess
from tgnn_fraud.data.synthetic import make_fixture
from tgnn_fraud.evaluation.run_eval import run_evaluation
from tgnn_fraud.explain.run_explain import export_alerts
from tgnn_fraud.training.train_baselines import train


def test_dashboard_queue_detail_and_status(tmp_path, monkeypatch):
    root = Path.cwd()
    config = yaml.safe_load((root / "configs/prototype.yaml").read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    config["data"]["input"] = "fixture.csv"
    config["data"]["output"] = "processed"
    config["runs_dir"] = "runs"
    config["baseline"]["n_estimators"] = 10
    make_fixture().to_csv("fixture.csv", index=False)
    preprocess(config)
    run = train(config)
    run_evaluation(run)
    export_alerts(run, config)
    app = AppTest.from_file(str(root / "app/Home.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert "label" not in app.dataframe[0].value.columns
    app.sidebar.checkbox[0].set_value(True).run()
    assert "label" in app.dataframe[0].value.columns
    app.sidebar.radio[0].set_value("Alert detail").run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any("saved" in item.value for item in app.success)


def test_dashboard_empty_setup(tmp_path, monkeypatch):
    root = Path.cwd()
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(root / "app/Home.py"), default_timeout=30).run()
    assert not app.exception
    assert "No saved alerts" in app.info[0].value


def test_dashboard_incomplete_run_is_useful_setup_message(tmp_path, monkeypatch):
    root = Path.cwd()
    monkeypatch.chdir(tmp_path)
    incomplete = tmp_path / "runs/incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "alerts.db").touch()
    app = AppTest.from_file(str(root / "app/Home.py"), default_timeout=30).run()
    assert not app.exception
    assert "No saved alerts" in app.info[0].value
    assert "Incomplete" in app.sidebar.warning[0].value
