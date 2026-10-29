import json
import numpy as np
import pandas as pd
import pytest
from src.stock_prediction import load_prices, prepare, evaluate, run


def test_chronology_scaler_and_windows():
    df = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=100), 'Close': np.arange(1., 101.)})
    scaler, splits, bounds = prepare(df, 10)
    assert bounds == (70, 85)
    assert scaler.data_max_[0] == 70
    assert splits['train'][2][-1] == 69
    assert splits['validation'][2].tolist() == list(range(70, 85))
    assert splits['test'][2].tolist() == list(range(85, 100))
    for x, y, idx in splits.values():
        np.testing.assert_allclose(scaler.inverse_transform(y).ravel(), df.Close.iloc[idx])
        for window, target in zip(x, idx):
            np.testing.assert_allclose(scaler.inverse_transform(window).ravel(), df.Close.iloc[target-10:target], rtol=1e-6)
    assert splits['test'][0].max() > 1  # Never refit/clip held-out data.


def test_known_metrics_and_flat_naive_direction():
    scores = evaluate([2, 4, 6], [1, 5, 5], [1, 3, 5])
    assert scores['MSE'] == scores['RMSE'] == scores['MAE'] == 1
    assert scores['R2'] == pytest.approx(.625)
    assert scores['DirectionalAccuracy'] == pytest.approx(1/3)
    assert evaluate([2, 4, 6], [1, 3, 5], [1, 3, 5])['DirectionalAccuracy'] == 0


@pytest.mark.parametrize('prices', [[1, np.nan, 3], [1, np.inf, 3], [1, 0, 3]])
def test_bad_prices_rejected(tmp_path, prices):
    path = tmp_path/'bad.csv'
    pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=3), 'Close': prices}).to_csv(path, index=False)
    with pytest.raises(ValueError): load_prices(path)


def test_duplicate_dates_rejected(tmp_path):
    path = tmp_path/'bad.csv'
    pd.DataFrame({'Date': ['2020-01-01']*2, 'Close': [1,2]}).to_csv(path, index=False)
    with pytest.raises(ValueError): load_prices(path)


def test_small_data_rejected():
    with pytest.raises(ValueError): prepare(pd.DataFrame({'Close': [1,2,3]}), 20)


def test_end_to_end_training_exports_and_reload(tmp_path):
    import tensorflow as tf
    metrics, predictions, meta = run(output=tmp_path, epochs=1)
    assert len(metrics) == 4 and np.isfinite(metrics.select_dtypes('number')).all().all()
    assert set(predictions.Split) == {'test', 'validation'}
    test = predictions.query("Split == 'test'")
    assert len(test) == meta['sample_counts']['test']
    np.testing.assert_allclose(test.Naive, test.PreviousClose)
    expected = evaluate(test.Actual, test.LSTM, test.PreviousClose)
    row = metrics.query("Split == 'test' and Model == 'LSTM'").iloc[0]
    for key in expected: assert row[key] == pytest.approx(expected[key])
    restored = tf.keras.models.load_model(tmp_path/'model.keras')
    df = load_prices('data/aapl_history.csv')
    scaler, splits, _ = prepare(df)
    actual = scaler.inverse_transform(restored(splits['test'][0], training=False).numpy()).ravel()
    np.testing.assert_allclose(actual, test.LSTM, rtol=1e-6)
    assert json.loads((tmp_path/'run_metadata.json').read_text())['completed_epochs'] == 1
    for name in ['training.png','data_split.png','test_predictions.png','residuals.png']:
        assert (tmp_path/name).stat().st_size > 1000
