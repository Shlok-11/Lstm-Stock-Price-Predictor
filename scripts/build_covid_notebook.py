"""Build an independent, standalone Colab experiment without clearing old outputs."""
from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parents[1]
pipeline=(root/'src/stock_prediction.py').read_text().split("if __name__ == '__main__':")[0]
experiment=(root/'experiments/covid_ablation.py').read_text().split("if __name__ == '__main__':")[0]
experiment=experiment.replace('from src.stock_prediction import build_model, dataset, evaluate, load_prices','')
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'COVID_Exclusion_Experiment.ipynb'}}
nb.cells=[md('''# Controlled COVID-era training ablation
This standalone notebook preserves the original 2015–2017 notebook and artifacts. Those observations predate COVID; the historical test results cannot answer this question.

**Status: real-data ablation not run in the cloud.** Bundled data ends in 2017; the proxy rejected Yahoo history with HTTP 403. TensorFlow training is available. Synthetic integration tests validate mechanics only; they are not stock performance results.

Run in a fresh Google Colab CPU runtime. No clone, Drive mount or credentials required. Download real AAPL data or upload a CSV with Date/Close. Do not substitute synthetic prices for reported performance.'''),
code('''import importlib.util
if importlib.util.find_spec('tensorflow') is None or importlib.util.find_spec('yfinance') is None:
    %pip install -q tensorflow-cpu==2.18.1 pandas==2.2.3 numpy==2.0.2 scikit-learn==1.6.1 matplotlib==3.10.1 yfinance==0.2.65
'''),
md('''## Predeclared protocol
A separate experiment: training 2015–2022, validation calendar 2023, test calendar 2024 (end exclusive 2025-01-01). This replaces neither the old dataset nor its original holdouts. All three regimes have exactly the same new validation/test target dates and inputs.

1. **with_2020**: all training years, same original LSTM architecture.
2. **exclude_2020**: omit every training window with ANY 2020 context or target row. Never splice the filtered series across the omitted year.
3. **post_2020**: train only on windows entirely within 2021–2022, requiring at least 252 training windows. This means post-2020, not a claim the pandemic ended in 2020.

Same 20-day lookback, architecture, optimizer, batch size, early-stopping budget and paired seeds 42/43/44. Fit a separate scaler only on each regime's eligible training rows. Pick checkpoints using validation loss; choose the regime by mean validation MSE in original price units. Save selection before testing. Report every fixed seed and its mean/std; do not pick the best seed. The test is evaluated once per frozen checkpoint. Do not change dates or settings after inspecting test outcomes.

Rolling one-step forecasts may use earlier realized held-out closes. No future target is in its own context. Source gaps over seven calendar days are rejected for investigation. Short missing sessions are not imputed; use complete daily trading data. Removing 2020 changes sample count and training distribution, so this is a sensitivity study, not causal proof of an effect of COVID volatility. Post-2020 may include pandemic effects. Adjusted prices reflect provider corporate-action revisions and are not a point-in-time data vintage.'''),
code(pipeline+'\n\n'+experiment),
code('''# Freeze these choices BEFORE training; use a NEW run directory each time.
PROTOCOL = Protocol()
CSV_PATH = None  # Set to an uploaded complete Date,Close CSV to avoid Yahoo.
DOWNLOAD_YAHOO = True
from datetime import datetime, timezone
OUTPUT = 'covid_runs/' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
print(asdict(PROTOCOL))
'''),
md('''## Real data acquisition and audit
Yahoo is optional and can rate limit. Failures are recorded explicitly without synthetic fallback. AAPL uses split/dividend-adjusted daily close consistently across the entire new experiment; never concatenate it with the original nominal CSV. A CSV upload should use a single consistent adjustment convention and asset. Record its provenance yourself. The run saves the input snapshot and SHA-256.'''),
code('''DATA_STATUS = 'not acquired'
if CSV_PATH is None and DOWNLOAD_YAHOO:
    try:
        import yfinance as yf
        raw = yf.download('AAPL', start=PROTOCOL.train_start, end=PROTOCOL.test_end,
                          auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            raise RuntimeError('Yahoo returned no observations')
        close = raw['Close']
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        folder = Path(OUTPUT)
        folder.parent.mkdir(parents=True, exist_ok=True)
        CSV_PATH = str(folder.parent / (folder.name + '_input.csv'))
        pd.DataFrame({'Date':close.index, 'Close':close.to_numpy()}).to_csv(CSV_PATH,index=False)
        Path(CSV_PATH + '.provenance.json').write_text(json.dumps({
            'provider':'Yahoo Finance via yfinance', 'ticker':'AAPL', 'auto_adjust':True,
            'retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'requested_start':PROTOCOL.train_start,'requested_end_exclusive':PROTOCOL.test_end,
            'sha256':hashlib.sha256(Path(CSV_PATH).read_bytes()).hexdigest()},indent=2))
    except Exception as error:
        CSV_PATH = None
        DATA_STATUS = f'BLOCKED: {type(error).__name__}: {error}'
if CSV_PATH:
    # Fail loudly for bad/insufficient user data before any fitting.
    audit_df, audit_regimes, skipped = prepare_experiment(load_prices(CSV_PATH), PROTOCOL)
    DATA_STATUS = 'Ready for one predeclared experiment'
    print({name:{split:len(value[2]) for split,value in splits.items()}
           for name,(_,splits) in audit_regimes.items()})
    print('Skipped:', skipped)
print(DATA_STATUS)
'''),
code('''if CSV_PATH:
    metrics, winner = run_experiment(CSV_PATH, OUTPUT, PROTOCOL)
    display(metrics)
    print('Validation-selected regime:', winner)
else:
    print('NOT RUN: provide real COVID-spanning data or retry on Colab with data access. No performance claims are available.')
'''),
md('''## Interpretation and export
MSE is price²; RMSE and MAE are price; R² is not accuracy. Directional accuracy compares sign(forecast − previous close) to sign(actual − previous close), treating zero separately. Naive previous-close forecasts predict zero movement, so their directional score is the unchanged-day frequency, not a 50% benchmark.

Compare every regime against the same naive forecasts. Use validation for selection, and test only as a final descriptive comparison. A lower error after dropping 2020 does not isolate volatility from reduced sample size, recency, price-level shifts, or seed variance. Three seeds quantify limited training variability, not a statistical significance test. If all variants lose to persistence, report it. Never copy synthetic test scores into the README as real-market evidence.

After a successful run, copy the printed real validation/test metrics and provenance to the README as a **new** results section, retaining the historical table. Archive output before any future study.'''),
code('''if CSV_PATH and Path(OUTPUT, 'metrics.csv').exists():
    import shutil
    archive = shutil.make_archive(OUTPUT, 'zip', OUTPUT)
    print('Saved', archive)
    try:
        from google.colab import files
        files.download(archive)
    except ImportError:
        pass
''')]
nbf.validate(nb)
nbf.write(nb,root/'notebooks/COVID_Exclusion_Experiment.ipynb')
print('Built standalone COVID notebook; historical notebook untouched')
