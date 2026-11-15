#!/usr/bin/env bash
# Runs every pipeline stage from a clean state and stores the console output of each one in
# reports/logs/console/ (these logs are rendered into the stage screenshots of the final report).
set -euo pipefail
cd "$(dirname "$0")/../.."
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS=ignore
PY=.venv/bin/python
OUT=reports/logs/console
mkdir -p "$OUT"
make clean >/dev/null

for s in ingest validate prepare transform features train evaluate; do
  { echo "\$ python -m src.pipeline --stage $s"; $PY -m src.pipeline --stage "$s" 2>&1; } > "$OUT/stage_$s.log"
  echo "stage $s ok"
done

{ echo '$ python -m src.predict --user-ids 1 945 --k 10'; $PY -m src.predict --user-ids 1 945 --k 10 2>&1; } > "$OUT/predict.log"
{ echo '$ python -m pytest -v'; $PY -m pytest -v -p no:cacheprovider 2>&1; } > "$OUT/tests.log"
{
  echo '$ wc -l data/errors/*.csv'; wc -l data/errors/*.csv
  echo; echo '$ cut -d, -f1-5,8 data/errors/interactions_errors.csv | head -20'
  cut -d, -f1-5,8 data/errors/interactions_errors.csv | head -20
} > "$OUT/errors_dataset.log"
{
  echo '$ cat reports/ingestion_manifest.json'; cat reports/ingestion_manifest.json
} > "$OUT/ingestion_manifest.log"
{
  echo '$ ls -R data/processed data/features | head -40'; ls -R data/processed data/features | head -40
} > "$OUT/data_layout.log"
