#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Lstm-Stock-Price-Predictor
python3 -m venv /workspace/venvs/lstm
/workspace/venvs/lstm/bin/python -m pip install -r requirements-lock.txt
/workspace/venvs/lstm/bin/python -m pip check
/workspace/venvs/lstm/bin/python -m ipykernel install --prefix /workspace/venvs/lstm --name lstm --display-name 'LSTM Python'
