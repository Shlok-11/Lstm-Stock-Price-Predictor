"""Chronological one-step stock forecasting. No test-set model selection."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_prices(path):
    df = pd.read_csv(path)
    df = df.rename(columns={'AAPL.Close': 'Close'})
    if not {'Date', 'Close'} <= set(df):
        raise ValueError('CSV must contain Date and Close columns')
    df = df[['Date', 'Close']].copy()
    df['Date'] = pd.to_datetime(df.Date, errors='raise')
    df['Close'] = pd.to_numeric(df.Close, errors='raise')
    if df.Date.duplicated().any() or df.isna().any().any():
        raise ValueError('Missing values and duplicate dates are not accepted')
    if not np.isfinite(df.Close).all() or (df.Close <= 0).any():
        raise ValueError('Prices must be finite and positive')
    return df.sort_values('Date').reset_index(drop=True)


def prepare(df, lookback=20):
    n = len(df)
    a, b = int(n * .7), int(n * .85)
    if lookback < 1 or a <= lookback or b-a < 2 or n-b < 2:
        raise ValueError('Insufficient observations for lookback and 70/15/15 split')
    values = df.Close.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler().fit(values[:a])
    scaled = scaler.transform(values).astype('float32')
    targets = np.arange(lookback, n)
    x = np.stack([scaled[i-lookback:i] for i in targets])
    y = scaled[targets]
    masks = {'train': targets < a, 'validation': (targets >= a) & (targets < b), 'test': targets >= b}
    splits = {name: (x[m], y[m], targets[m]) for name, m in masks.items()}
    return scaler, splits, (a, b)


def evaluate(actual, predicted, previous):
    actual, predicted, previous = map(lambda v: np.asarray(v).reshape(-1), (actual, predicted, previous))
    if not (len(actual) == len(predicted) == len(previous)) or len(actual) < 2:
        raise ValueError('Metrics require aligned arrays with at least two rows')
    if not all(np.isfinite(v).all() for v in (actual, predicted, previous)):
        raise ValueError('Metrics require finite values')
    mse = mean_squared_error(actual, predicted)
    return {'MSE': float(mse), 'RMSE': float(np.sqrt(mse)),
            'MAE': float(mean_absolute_error(actual, predicted)),
            'R2': float(r2_score(actual, predicted)),
            'DirectionalAccuracy': float(np.mean(np.sign(actual-previous) == np.sign(predicted-previous)))}


def build_model(lookback):
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.Input((lookback, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mse')
    return model


def dataset(x, y, batch_size=32):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return ds.with_options(options)


def run(csv='data/aapl_history.csv', output='artifacts', epochs=40, lookback=20, ticker='AAPL', seed=42):
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if epochs < 1:
        raise ValueError('epochs must be positive')
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    df = load_prices(csv)
    scaler, splits, bounds = prepare(df, lookback)
    model = build_model(lookback)
    train, val = splits['train'], splits['validation']
    history = model.fit(dataset(*train[:2]), validation_data=dataset(*val[:2]), epochs=epochs, shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)], verbose=0)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows = [], []
    for split in ('validation', 'test'):
        x, _, idx = splits[split]
        prediction = scaler.inverse_transform(model(x, training=False).numpy()).ravel()
        actual = df.Close.to_numpy()[idx]
        previous = df.Close.to_numpy()[idx-1]
        for name, pred in [('LSTM', prediction), ('Naive', previous)]:
            metric_rows.append({'Split': split, 'Model': name, **evaluate(actual, pred, previous)})
        prediction_rows.append(pd.DataFrame({'Date': df.Date.iloc[idx].dt.strftime('%Y-%m-%d').to_numpy(),
            'Ticker': ticker, 'Split': split, 'Actual': actual, 'PreviousClose': previous,
            'LSTM': prediction, 'Naive': previous,
            'LSTMAbsoluteError': abs(actual-prediction), 'NaiveAbsoluteError': abs(actual-previous),
            'ActualDirection': np.sign(actual-previous), 'LSTMDirection': np.sign(prediction-previous),
            'NaiveDirection': np.zeros(len(idx), dtype=int)}))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(out/'metrics.csv', index=False)
    predictions.to_csv(out/'power_bi_predictions.csv', index=False)
    pd.DataFrame(history.history).rename_axis('EpochZeroBased').to_csv(out/'training_history.csv')
    model.save(out/'model.keras')
    (out/'scaler.json').write_text(json.dumps({'min': scaler.min_.tolist(), 'scale': scaler.scale_.tolist(),
        'data_min': scaler.data_min_.tolist(), 'data_max': scaler.data_max_.tolist()} , indent=2))
    meta = {'ticker': ticker, 'seed': seed, 'lookback': lookback, 'requested_epochs': epochs,
        'completed_epochs': len(history.history['loss']), 'best_epoch': int(np.argmin(history.history['val_loss'])+1),
        'rows': len(df), 'split_boundaries': list(bounds),
        'date_range': [str(df.Date.min().date()), str(df.Date.max().date())],
        'sample_counts': {k: len(v[0]) for k,v in splits.items()},
        'data_sha256': hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
        'tensorflow': tf.__version__, 'numpy': np.__version__,
        'protocol': 'Rolling one-step forecasts using observed previous closes; no recursive forecasts.'}
    (out/'run_metadata.json').write_text(json.dumps(meta, indent=2))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.Date, df.Close, label='Close')
    for i, label in zip(bounds, ['Validation starts', 'Test starts']): ax.axvline(df.Date.iloc[i], linestyle='--', label=label)
    ax.set(title=f'{ticker}: chronological split', ylabel='Price (historical USD)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'data_split.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    for k in ['loss','val_loss']: ax.plot(np.arange(1,len(history.history[k])+1),history.history[k], label=k)
    ax.set(xlabel='Epoch', ylabel='Scaled MSE', title='Learning curves'); ax.legend(); fig.tight_layout(); fig.savefig(out/'training.png'); plt.close(fig)
    test = predictions[predictions.Split == 'test']
    fig, ax = plt.subplots(figsize=(10,4))
    for k in ['Actual','LSTM','Naive']: ax.plot(pd.to_datetime(test.Date),test[k], label=k)
    ax.set(title='Untouched test period: one-step predictions', ylabel='Price (historical USD)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'test_predictions.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    for k in ['LSTM','Naive']: ax.hist(test.Actual-test[k], bins=20, alpha=.5, label=k)
    ax.set(title='Test residuals', xlabel='Actual minus predicted (USD)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'residuals.png'); plt.close(fig)
    print(metrics.to_string(index=False))
    return metrics, predictions, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/aapl_history.csv')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lookback', type=int, default=20)
    parser.add_argument('--ticker', default='AAPL')
    args = parser.parse_args()
    run(**vars(args))
