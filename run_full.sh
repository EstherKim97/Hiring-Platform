#!/usr/bin/env bash
set -euo pipefail

echo "Running full pipeline. Ensure you've installed optional dependencies: pip install -r requirements-optional.txt"

python3 src/load_data.py || true
python3 src/run_experiment.py
python3 src/analyze.py
python3 src/charts.py

echo "Full pipeline complete"
