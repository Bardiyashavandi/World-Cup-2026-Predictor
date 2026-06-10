#!/usr/bin/env bash
#
# run_all.sh — runs the full World Cup 2026 prediction pipeline end to end.
#
# Usage:
#   ./run_all.sh            # process -> features -> all models -> ensemble
#   ./run_all.sh --backtest # also run the backtest at the end
#
# Stops immediately if any step fails.

set -euo pipefail

cd "$(dirname "$0")"

echo "=============================================="
echo "  World Cup 2026 Predictor — full pipeline"
echo "=============================================="

echo ""
echo ">> [1/5] Processing & cleaning data..."
python3 src/data/process_data.py

echo ""
echo ">> [2/5] Engineering features..."
python3 src/features/build_features.py

echo ""
echo ">> [3/5] Running baseline models..."
python3 src/models/baseline/elo_model.py
python3 src/models/baseline/dixon_coles.py
python3 src/models/baseline/historical_avg.py
python3 src/models/baseline/dynamic_elo.py

echo ""
echo ">> [4/5] Running ML models..."
python3 src/models/ml/xgboost_model.py
python3 src/models/ml/lightgbm_model.py
python3 src/models/ml/neural_network.py
python3 src/models/ml/logistic_regression.py

echo ""
echo ">> [5/5] Generating ensemble predictions (MD1-3)..."
python3 src/ensemble/ensemble.py 1
python3 src/ensemble/ensemble.py 2
python3 src/ensemble/ensemble.py 3

if [[ "${1:-}" == "--backtest" ]]; then
  echo ""
  echo ">> Running backtest..."
  python3 src/evaluation/backtest.py
fi

echo ""
echo "=============================================="
echo "  Pipeline complete."
echo "  Launch the dashboard with:"
echo "    streamlit run dashboard/app.py"
echo "=============================================="
