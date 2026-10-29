# COVID training ablation

The runnable implementation is `covid_ablation.py`; the standalone Colab notebook embeds it and the unchanged model functions. Rebuild only the new notebook with `python scripts/build_covid_notebook.py` (this clears its outputs; execute it afterward to record status). The original notebook generator does not generate or execute the COVID study.

Current verified state: market-data experiment **not run**. The repository's frozen CSV ends in 2017. Both a requests chart probe and the notebook's yfinance download encountered the configured proxy's HTTP 403 denial. Training is functional, as verified by the synthetic integration test; no test score from synthetic data is a market result. No regime can be selected yet. See README for the frozen proposed dates and controls.

Run all Colab cells once using a real complete AAPL CSV covering 2015–2024, or the default Yahoo download when access permits. The downloader records adjustment convention, retrieval timestamp, source and checksum. For a supplied CSV, record provider, retrieval date, asset and adjustment convention alongside it. Do not concatenate nominal and adjusted prices.

Retain the output folder and its `protocol.json`, `selection.json`, models, histories, scaler information, predictions, input snapshot and metric tables. If execution fails, keep its partial directory for diagnosis; a rerun must use a new directory. Do not tune protocol choices using test scores, even if the validation winner loses to the naive baseline. End-to-end tests use temporary folders and do not overwrite existing historical results.
