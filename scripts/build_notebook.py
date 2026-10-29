"""Generate a standalone Colab notebook from the tested pipeline and frozen data."""
from pathlib import Path
import base64
import gzip
import nbformat as nbf
root = Path(__file__).resolve().parents[1]
code = (root/'src/stock_prediction.py').read_text().split("if __name__ == '__main__':")[0]
data = base64.b64encode(gzip.compress((root/'data/aapl_history.csv').read_bytes(), mtime=0)).decode()
nb = nbf.v4.new_notebook()
nb.metadata = {'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}, 'colab': {'name':'LSTM_Stock_Prediction.ipynb'}}
md, cell = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md('''# LSTM Stock Price Prediction
A standalone, executable Google Colab notebook. Run all cells in order on a CPU runtime. The included real AAPL snapshot permits an offline experiment after installing packages. Optional Yahoo data is an explicit alternative, never a silent fallback.

Inspired by [Syed-Waheed/stock-price-prediction-lstm](https://github.com/Syed-Waheed/stock-price-prediction-lstm). This implementation uses smaller stacked LSTMs and a separate validation period. Educational use; no investment advice.'''),
cell('''# Install only if the pinned workflow is not already available (e.g. fresh Colab).
import importlib.util
if importlib.util.find_spec('tensorflow') is None or importlib.util.find_spec('yfinance') is None:
    %pip install -q tensorflow-cpu==2.18.1 pandas==2.2.3 numpy==2.0.2 scikit-learn==1.6.1 matplotlib==3.10.1 yfinance==0.2.65
'''),
md('''## 1. Historical data and provenance
The embedded CSV is the unmodified `finance-charts-apple.csv` from https://github.com/plotly/datasets (downloaded 2026-10-29). We use `AAPL.Close`, the historical nominal USD close, not the adjusted column. This is a small 2015–2017 demonstration, not current market evidence. Corporate actions, data revisions and data-use terms matter for other datasets.'''),
cell("import base64, gzip\nfrom pathlib import Path\nPath('data').mkdir(exist_ok=True)\nPath('data/aapl_history.csv').write_bytes(gzip.decompress(base64.b64decode("+repr(data)+")))\nCSV = 'data/aapl_history.csv'\nTICKER = 'AAPL'"),
md('''## 2. Optional Yahoo Finance data
Leave disabled to reproduce the included experiment. Yahoo requires network access and may rate limit requests. Adjusted Yahoo prices differ from the embedded nominal prices. Changing data defines a new experiment; do not compare its scores as if it were the same run.'''),
cell('''USE_YAHOO = False
if USE_YAHOO:
    import yfinance as yf
    import pandas as pd
    TICKER = 'AAPL'  # e.g. 'RELIANCE.NS'; change units in plots/docs accordingly
    raw = yf.download(TICKER, start='2015-01-01', end='2025-01-01', auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError('Yahoo returned no data; check access/ticker; no fallback was used.')
    close = raw['Close']
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    pd.DataFrame({'Date': close.index, 'Close': close.to_numpy()}).to_csv('data/yahoo.csv', index=False)
    CSV = 'data/yahoo.csv'
'''),
md('''## 3. Method and reusable pipeline
Predict the next observed trading day's close from the previous 20 observed closes. Chronological raw-row splits are 70% training, 15% validation and 15% test. Fit MinMaxScaler **only** on training rows; held-out values may lie outside [0,1]. Windows never include their target. Validation/test windows may use earlier observed closes, including earlier held-out dates: this is rolling one-step evaluation, not a forecast of the whole test period from a single origin.

Architecture: LSTM(32, return_sequences=True) → LSTM(16) → Dense(1), Adam and scaled MSE. Early stopping selects weights using validation loss only; test data is untouched until final evaluation. Fixed seed and deterministic TensorFlow operations aid reproducibility but hardware/library changes can change results.'''),
cell(code),
md('''## 4. Train once, evaluate, and export
The naive baseline predicts tomorrow's price equals today's price. Report MSE (price²), RMSE and MAE (price), R² (can be negative), and directional accuracy (fraction in [0,1]). Direction compares sign(predicted − previous close) with sign(actual − previous close), including zero as a distinct outcome. A flat naive forecast therefore scores only on unchanged days. High R² on price levels is not trading profitability. Test scores must not guide repeated tuning.'''),
cell("metrics, predictions, metadata = run(CSV, output='artifacts', epochs=40, lookback=20, ticker=TICKER)\ndisplay(metrics)\ndisplay(predictions.head())\nprint(metadata)"),
md('## 5. Graphs: split, learning curves, test forecasts, residuals'),
cell("from IPython.display import Image, display\nfor name in ['data_split', 'training', 'test_predictions', 'residuals']:\n    display(Image(filename=f'artifacts/{name}.png'))"),
md('''## 6. Power BI and downloads
Import `power_bi_predictions.csv` with Date as Date and prices as Decimal Number. Filter Split to test. Use Date as a continuous line axis and Actual/LSTM/Naive as values. Import `metrics.csv` separately and filter Split and Model for metric cards; do not sum per-run metrics. DirectionalAccuracy is a fraction; format as percent. No PBIX file is required. The export contains observed test targets: it is a retrospective evaluation dataset.

The ZIP includes the saved Keras model, scaler parameters, run metadata with data checksum, training history, metrics, predictions and four PNG graphs.'''),
cell("import shutil\nshutil.make_archive('lstm_results', 'zip', 'artifacts')\ntry:\n    from google.colab import files\n    files.download('lstm_results.zip')\nexcept ImportError:\n    print('Saved lstm_results.zip in the current directory.')"),
md('''## 7. Interpretation and limitations
Compare test RMSE/MAE with persistence before claiming value. If LSTM loses, report that result. A small single-stock sample, regime changes, price-level nonstationarity, corporate-action handling and seed variability limit generalization. No transaction costs, execution simulation, exogenous features or production monitoring are modeled. Future work: walk-forward evaluation, training-only tuning, returns targets, multiple seeds and cost-aware backtests. None is evidence of profitability until measured.

### Interview questions
1. **Why LSTM?** Its gates can retain temporal information; this does not guarantee an advantage over persistence.
2. **Why no shuffled split?** Future information must not influence earlier forecasts.
3. **Where is scaling fitted?** Training rows only; validation/test are transformed without refitting.
4. **What does return_sequences do?** It passes one hidden output per time step to the next LSTM layer.
5. **Why separate validation and test?** Validation chooses stopping weights; the test estimates held-out performance once.
6. **Why a naive baseline?** Stock price levels are persistent; a sophisticated model must demonstrate added value.
7. **Is R² accuracy?** No. It measures squared-error reduction relative to a constant mean, not a percentage of correct forecasts.
8. **How is direction measured?** Compare both predicted and actual changes against the same previous observed close.
9. **Does prior test context leak?** Observed earlier closes are valid for rolling one-step forecasts, but not for multi-step forecasts made at the start of the test period.
10. **Why can a scaled test value exceed one?** Prices can exceed the training range; clipping would conceal that information.
11. **How does early stopping help?** It restores the validation-best weights and reduces unnecessary fitting, without guaranteeing generalization.
12. **Can this be traded directly?** No; predictive evaluation omits costs, risk constraints, execution delays and robust backtesting.''')]
nbf.validate(nb)
nbf.write(nb, root/'notebooks/LSTM_Stock_Prediction.ipynb')
