PY := .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: setup pipeline flow serve test predict mlflow-ui prefect-ui clean

setup:            ## create the virtual environment and install pinned dependencies
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

pipeline:         ## run every stage in order (plain Python runner)
	$(PY) -m src.pipeline

flow:             ## run the Prefect flow once (start `make prefect-ui` first to see it in the UI)
	PREFECT_API_URL=http://127.0.0.1:4200/api $(PY) dags/recommendation_flow.py

serve:            ## scheduled Prefect deployment (weekly, Monday 02:00)
	PREFECT_API_URL=http://127.0.0.1:4200/api $(PY) dags/recommendation_flow.py --serve

test:
	$(PY) -m pytest -q

predict:
	$(PY) -m src.predict --user-ids 1 13 945 --k 10

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///models/mlflow.db --port 5000

prefect-ui:
	.venv/bin/prefect server start --host 127.0.0.1 --port 4200

clean:            ## remove all generated artefacts (keeps the downloaded archive)
	find data -type f ! -name ml-100k.zip ! -name .gitkeep -delete; rm -rf models/*.pkl models/mlruns models/mlflow.db
