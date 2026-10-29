# Historical data

`aapl_history.csv` is the unchanged Plotly `finance-charts-apple.csv` public sample. Source URL, retrieval date, selected price field and SHA-256 are in `provenance.json`. The pipeline uses Date and AAPL.Close only. It does not infer missing dates or silently impute prices. Corporate-action-adjusted data is preferable for wider periods crossing splits; this bundled nominal-price sample must not be mixed with adjusted Yahoo prices in a single experiment.

For another stock, supply a CSV with Date and Close columns and pass `--csv` and `--ticker`. Dates are sorted; duplicates, missing values, nonfinite values and nonpositive prices fail explicitly. The notebook has an optional Yahoo download cell; access is not needed for the included frozen experiment. Review vendor terms before redistribution/commercial use.
