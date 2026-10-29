# Power BI report

Use Get data → Text/CSV for `artifacts/power_bi_predictions.csv` and `artifacts/metrics.csv`.
Set Date to Date, Ticker/Split/Model to Text, prices/errors/metrics to Decimal Number, and direction fields to Whole Number.

- Add a Split slicer and select **test** for final reporting.
- Line chart: Date on a continuous X axis, Actual, LSTM and Naive on Y. Disable automatic date hierarchy if it aggregates days.
- Error chart: Date against LSTMAbsoluteError and NaiveAbsoluteError.
- Metric cards from metrics: filter exactly one Split and Model, use MAX (one precomputed row), never SUM across runs or models.
- Format DirectionalAccuracy as Percentage. R2 is a coefficient, not prediction accuracy.
- Keep the metric table disconnected or use a separate Split dimension linked to both tables; do not create a many-to-many join on Split.

Optional measures calculated from prediction rows (respect Split filters):

```dax
LSTM MSE = AVERAGEX(power_bi_predictions, POWER(power_bi_predictions[Actual] - power_bi_predictions[LSTM], 2))
LSTM RMSE = SQRT([LSTM MSE])
LSTM MAE = AVERAGE(power_bi_predictions[LSTMAbsoluteError])
LSTM Directional Accuracy = AVERAGEX(power_bi_predictions, IF(power_bi_predictions[ActualDirection] = power_bi_predictions[LSTMDirection], 1.0, 0.0))
```

Naive predicts PreviousClose. Its predicted direction is always zero, and strict three-way directional accuracy counts only truly unchanged closes as correct. These are retrospective rolling one-step forecasts, not future prices or a trading signal. The bundled prices are historical nominal USD values. No Power BI Desktop runtime is installed here, so a PBIX report is not claimed or tested.
