# LSTM Stock Price Prediction

An executable Google Colab notebook and tested Python pipeline for rolling one-step stock closing-price prediction. It uses a simple stacked LSTM, real historical data, training-only scaling, chronological train/validation/test splits, and a persistence baseline. **The measured LSTM loses to the naive baseline on price error in the included experiment.** This is an educational project, not a trading system.

## Open the notebook

Upload [`notebooks/LSTM_Stock_Prediction.ipynb`](notebooks/LSTM_Stock_Prediction.ipynb) to [Google Colab](https://colab.research.google.com/) and choose Run all on a CPU runtime. It is standalone: pipeline code and a compressed real historical CSV are embedded. No repository clone, credentials, GPU or Drive mount is required. Downloads include graphs, CSVs, model, scaler and metadata. Notebook outputs in this checkout were executed in the cloud Python environment, not the hosted Google Colab service.

The first cell installs missing ML dependencies. An existing Colab runtime can have different preinstalled versions; for exact dependency reproduction use the Python 3.12 environment below. If pip changes already-imported packages, restart the Colab runtime before Run all. Optional Yahoo Finance download is disabled by default and requires Internet access; its availability was not validated here. Failures are explicit and never substituted with synthetic data.

## Local / cloud setup

Python 3.12 on Linux CPU was validated. In this cloud checkout:

```bash
bash scripts/install_environment.sh
source /workspace/venvs/lstm/bin/activate
python -m pytest -q
python -m src.stock_prediction --epochs 40
JUPYTER_PATH=/workspace/venvs/lstm/share/jupyter python scripts/execute_notebook.py
```

On another machine, create a venv at your preferred path and `python -m pip install -r requirements-lock.txt`. `requirements.txt` records direct dependencies; `requirements-lock.txt` freezes the validated environment including transitives. TensorFlow CPU is sufficient. Notebook execution via the helper expects a kernel named `lstm`; install it with `python -m ipykernel install --prefix "$VIRTUAL_ENV" --name lstm` and set JUPYTER_PATH to that venv's share/jupyter directory. Run commands from the project root. Existing cloud tasks are isolated; use this checkout without creating a worktree.

To use your own data:

```bash
python -m src.stock_prediction --csv path/to/prices.csv --ticker TCS.NS --lookback 20 --epochs 40 --output artifacts/custom
```

CSV columns: Date, Close; prices must be positive and finite, dates unique. Plot labels default to historical USD for the bundled AAPL experiment; update units for other currencies. A Yahoo example using adjusted prices is available in the notebook. Never combine adjusted and nominal close series.

## Method

- Dataset: 506 real AAPL observations, 2015-02-17 through 2017-02-16, from Plotly's public historical sample. [Provenance and SHA-256](data/provenance.json); original CSV retained unchanged.
- Chronological raw-row boundaries: 354 training, 76 validation, 76 test. A 20-session lookback yields 334 training samples and 76 each for validation/test.
- Fit MinMaxScaler only on training rows; transform validation/test without refitting or clipping. No target appears in its input window.
- Architecture: LSTM(32, return_sequences=True), LSTM(16), Dense(1). Adam, learning rate 0.001, scaled MSE, batch size 32, ordered samples.
- Train up to 40 epochs with validation-only early stopping, patience 7, restored best weights. The recorded run stopped after 13 epochs and selected epoch 6. Seed 42 and deterministic TensorFlow operations enabled.
- Windows in validation/test use earlier realized closes as they become available. This is a rolling one-step evaluation, not a recursive multi-day forecast from the beginning of the test period.
- Inverse-transform forecasts before scoring. Baseline predicts the previous observed closing price. The same dates and targets are used for both models. The held-out test set is not used for model selection.

## Actual results

Full notebook run on the bundled data, TensorFlow 2.18.1 / NumPy 2.0.2, CPU:

| Split | Model | MSE | RMSE | MAE | R² | Directional accuracy |
|---|---|---:|---:|---:|---:|---:|
| Validation | LSTM | 8.353565 | 2.890253 | 2.447223 | 0.753001 | 48.6842% |
| Validation | Naive | 1.987980 | 1.409957 | 0.939737 | 0.941219 | 0.0000% |
| Test | LSTM | 29.615397 | 5.442003 | 4.271295 | 0.480147 | 40.7895% |
| Test | Naive | 1.729928 | 1.315267 | 0.820395 | 0.969634 | 1.3158% |

MSE is in historical USD²; RMSE and MAE are historical USD. R² is a coefficient, not accuracy. Directional accuracy compares sign(prediction − previous close) with sign(actual − previous close), counting zero as a distinct direction. Persistence always forecasts zero change, so its directional score measures the fraction of unchanged days; it is not a random 50% directional benchmark. The LSTM's higher directional score does not establish trading value and does not offset its much larger price errors.

These results are a small historical demonstration, not evidence about current markets. No hyperparameter tuning against the test set was performed. Generalization, transaction costs, slippage, corporate actions, regime shifts, seed sensitivity and robust walk-forward backtests remain limitations.

## Deliverables

- [Executed Colab notebook](notebooks/LSTM_Stock_Prediction.ipynb), [pipeline](src/stock_prediction.py), [tests](tests/test_pipeline.py).
- [Power BI predictions](artifacts/power_bi_predictions.csv), [metrics](artifacts/metrics.csv), [training history](artifacts/training_history.csv), [run metadata](artifacts/run_metadata.json), [scaler](artifacts/scaler.json).
- [Split graph](artifacts/data_split.png), [learning curves](artifacts/training.png), [test predictions](artifacts/test_predictions.png), [residuals](artifacts/residuals.png).
- [Results ZIP](lstm_results.zip), including the trained Keras model. Loose `*.keras` files are ignored by Git but generated on each run.
- [Power BI instructions and DAX](docs/POWER_BI.md). Import the CSVs in Power BI Desktop; no PBIX file or Power BI runtime is claimed.
- [20 interview questions with answers](docs/INTERVIEW_QA.md).

## Validation

`python -m pytest -q`: **8 passed**, including chronological indexing, training-only scaler fit, known metric values, invalid-data rejection, full one-epoch training/export and saved-model reload equivalence. Third-party Matplotlib/Pyparsing deprecation warnings do not fail tests. The full notebook separately ran all code cells without errors and produced the reported 40-epoch-budget experiment. `pip check` passed. Optional live Yahoo and hosted Colab/Power BI service behavior were not tested.

Regenerate the notebook from the tested source with `python scripts/build_notebook.py`; this clears its stored outputs, so rerun `scripts/execute_notebook.py` afterward. Repeated execution replaces artifacts with the new experiment; preserve results before changing configuration.

## Reference and attribution

Inspired by [Syed-Waheed/stock-price-prediction-lstm](https://github.com/Syed-Waheed/stock-price-prediction-lstm), inspected at commit `6ef969243d33e40077304fcb72b75e39c8d78939`. Architecture and workflow were independently implemented for this project; reference performance claims and trained models were not reused. The reference's [MIT license notice](docs/REFERENCE_LICENSE.txt) is retained for attribution. Dataset provenance is documented separately.
