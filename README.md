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

## Controlled COVID-era ablation — prepared, real-data run blocked

The existing dataset has **506 observations from 2015-02-17 through 2017-02-16 and no 2020 data**. The historical results above predate COVID and cannot establish a COVID-volatility effect. They, their dataset, plots, CSVs, metadata and ZIP have been preserved. No COVID model scores or winning regime have been measured or inferred.

On 2026-10-29, this cloud environment's configured proxy rejected real AAPL history from `query1.finance.yahoo.com` with **HTTP 403 CONNECT**. The standalone notebook's yfinance request also failed with a 403 tunnel response. **Data access is blocked; TensorFlow training works.** No synthetic prices were used as substitute market evidence.

Open the new standalone [`COVID_Exclusion_Experiment.ipynb`](notebooks/COVID_Exclusion_Experiment.ipynb) in Google Colab and Run all on a CPU runtime with Internet access. It downloads a single consistent Yahoo-adjusted AAPL series; alternatively upload a complete, consistently adjusted `Date,Close` CSV and set `CSV_PATH` in its configuration cell. The original notebook remains the historical demonstration, with all its existing executed cells preserved. Hosted Colab itself has not been tested.

The new experiment uses a **separate predeclared schedule**, because the original 2016–2017 holdouts occur before the intervention:

| Variant | Eligible training observations | Validation | Test |
|---|---|---|---|
| `with_2020` | 2015–2022, including 2020 | 2023 calendar year | 2024 calendar year |
| `exclude_2020` | 2015–2022, excluding every window touching 2020 | Same dates | Same dates |
| `post_2020` | 2021–2022 only, if ≥252 training windows | Same dates | Same dates |

`with_2020` uses the original **architecture**, not the historical fitted model: LSTM(32, return_sequences=True) → LSTM(16) → Dense(1). “Post-2020” is an operational date definition, not a claim that COVID ended in 2020. The original dataset and historical metrics must not be mixed with this new adjusted-price experiment.

Controls and outputs:

- Identical 20-session lookback, Adam 0.001, batch size 32, ordered samples, 40-epoch budget, patience 7 and paired seeds 42/43/44 across regimes. No test-driven tuning.
- Fit each scaler using only that regime's eligible training observations. Construct windows against the original timeline; every training context **and target** must be eligible. No December-2019/January-2021 splice or future target in its context. Reject source outages longer than seven calendar days for investigation; require complete daily trading data and do not impute missing prices.
- All regimes use exactly the same validation/test targets and rolling one-step inputs. Earlier realized holdout closes are valid context. No recursive multi-day forecast is claimed.
- Restore checkpoints by validation loss. Choose the training regime using **mean validation MSE in price units across all three fixed seeds**, then persist `selection.json` before test prediction. Report every seed rather than selecting the best seed.
- Compare MSE, RMSE, MAE, R² and directional accuracy against identical previous-close forecasts. Export per-seed scores and mean/std, predictions, input snapshot/checksum, training histories, scaler parameters, models and validation selection. Directional accuracy retains the historical zero-change convention described above.
- Each invocation requires a new output directory and refuses to overwrite earlier results. Existing historical artifacts are never used as the output destination.

For a real local CSV in the pinned environment:

```bash
source /workspace/venvs/lstm/bin/activate
python -m experiments.covid_ablation --csv path/to/aapl_2015_2024.csv --output artifacts/covid/run_001
```

**Verified checks:** `python -m pytest -q` → **14 passed** (8 existing + 6 new), including three-regime one-epoch synthetic training/export, identical holdout dates, no windows crossing the excluded year, training-only scaling, validation selection, insufficient-data handling and overwrite protection. Synthetic test scores are not financial results. `uv pip check` passed for the repository's pinned environment. All six code cells of the new standalone notebook executed locally and correctly recorded **NOT RUN** for the blocked market experiment. Existing historical experiment outputs were not regenerated.

**Interpretation after a real run:** removal of 2020 changes training size and distribution as well as exposure to volatility. A performance difference is sensitivity to the training regime, not causal proof of COVID's impact. Post-2020 data may still include pandemic effects. Three seeds do not establish statistical significance; adjusted historical prices are not a point-in-time vintage. Report losses to persistence openly. Do not change this protocol after observing test scores. Once the real-data run completes, append its verified metrics and provenance here without replacing the historical table.
