PYTHON ?= python
RUN ?=
CONFIG ?= configs/prototype.yaml
.PHONY: setup fixture data train-baselines train-tgn evaluate explain demo-bundle compare app test lint
setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
fixture:
	$(PYTHON) -m tgnn_fraud.data.synthetic --output data/raw/fixture.csv
data:
	$(PYTHON) -m tgnn_fraud.data.preprocess --config $(CONFIG)
train-baselines:
	$(PYTHON) -m tgnn_fraud.training.train_baselines --config $(CONFIG)
train-tgn:
	$(PYTHON) -m tgnn_fraud.training.train_tgn --config $(CONFIG)
evaluate:
	$(PYTHON) -m tgnn_fraud.evaluation.run_eval --run $(RUN)
explain:
	$(PYTHON) -m tgnn_fraud.explain.run_explain --run $(RUN) --config $(CONFIG)
demo-bundle:
	$(PYTHON) -m tgnn_fraud.serving.build_demo_bundle --run $(RUN) --config $(CONFIG)
compare:
	$(PYTHON) -m tgnn_fraud.evaluation.compare --runs $(RUNS)
app:
	$(PYTHON) -m streamlit run app/Home.py
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
