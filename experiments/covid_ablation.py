"""Predeclared COVID-era ablation; never overwrites the historical experiment."""
import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.stock_prediction import build_model, dataset, evaluate, load_prices


@dataclass(frozen=True)
class Protocol:
    train_start: str = '2015-01-01'
    validation_start: str = '2023-01-01'
    test_start: str = '2024-01-01'
    test_end: str = '2025-01-01'  # exclusive
    post_start: str = '2021-01-01'  # post-2020, not a claim that COVID ended
    lookback: int = 20
    epochs: int = 40
    patience: int = 7
    seeds: tuple = (42, 43, 44)
    minimum_train_windows: int = 252


def prepare_experiment(df, p=Protocol()):
    dates = df.Date
    start, val, test, end = map(pd.Timestamp, (p.train_start, p.validation_start, p.test_start, p.test_end))
    if not start < val < test < end or not start < pd.Timestamp(p.post_start) < val:
        raise ValueError('Invalid chronological date boundaries')
    if p.lookback < 1 or p.epochs < 1 or p.patience < 1 or p.minimum_train_windows < 1:
        raise ValueError('Invalid training parameters')
    if not p.seeds or len(set(p.seeds)) != len(p.seeds):
        raise ValueError('Seeds must be nonempty and unique')
    df = df.loc[(dates >= start) & (dates < end)].reset_index(drop=True)
    if df.empty or not df.Date.is_monotonic_increasing or df.Date.duplicated().any():
        raise ValueError('Require sorted, unique observations')
    dates = df.Date
    # Refuse unexplained long source outages rather than silently joining them.
    if dates.diff().dt.days.dropna().gt(7).any():
        raise ValueError('Source gap over seven calendar days; supply complete daily trading history')
    if not (dates.dt.year == 2020).any() or not (dates < '2020-01-01').any():
        raise ValueError('COVID comparison requires pre-2020 and 2020 observations; bundled 2015–2017 data is insufficient')
    targets = np.arange(p.lookback, len(df))
    held = {
        'validation': targets[(dates.iloc[targets] >= val) & (dates.iloc[targets] < test)],
        'test': targets[(dates.iloc[targets] >= test) & (dates.iloc[targets] < end)],
    }
    for name, idx in held.items():
        if len(idx) < 2:
            raise ValueError(f'Insufficient {name} observations for fixed dates')
    if dates.min() > start + pd.Timedelta(days=7) or dates.max() < end - pd.Timedelta(days=7):
        raise ValueError('Data does not cover the predeclared experiment date range')
    training = dates < val
    masks = {
        'with_2020': training.to_numpy(),
        'exclude_2020': (training & (dates.dt.year != 2020)).to_numpy(),
        'post_2020': (training & (dates >= p.post_start)).to_numpy(),
    }
    values = df.Close.to_numpy().reshape(-1, 1)
    prepared, skipped = {}, {}
    for name, allowed in masks.items():
        # Build against original row positions, then require EVERY context/target row.
        # Filtering rows first would splice December 2019 to January 2021.
        idx = np.array([i for i in targets if allowed[i-p.lookback:i+1].all()], dtype=int)
        if len(idx) < p.minimum_train_windows:
            if name != 'post_2020':
                raise ValueError(f'{name}: need {p.minimum_train_windows} training windows, got {len(idx)}')
            skipped[name] = f'Only {len(idx)} training windows; minimum {p.minimum_train_windows}'
            continue
        scaler = MinMaxScaler().fit(values[allowed])
        scaled = scaler.transform(values).astype('float32')
        splits = {}
        for split, indices in {'train': idx, **held}.items():
            x = np.stack([scaled[i-p.lookback:i] for i in indices])
            splits[split] = (x, scaled[indices], indices)
        prepared[name] = (scaler, splits)
    return df, prepared, skipped


def run_experiment(csv, output, p=Protocol()):
    import tensorflow as tf
    df, prepared, skipped = prepare_experiment(load_prices(csv), p)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)  # refuse replacing ANY earlier run
    protocol = {**asdict(p), 'data_sha256': hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
                'tensorflow': tf.__version__, 'numpy': np.__version__, 'skipped': skipped,
                'selection': 'Lowest mean validation MSE in price units across fixed seeds; no test selection',
                'evaluation': 'Rolling one-step observed context; fixed validation/test dates for all variants',
                'architecture': 'LSTM(32,return_sequences=True),LSTM(16),Dense(1); Adam 0.001; batch 32; shuffle False'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    df.to_csv(out/'input_snapshot.csv', index=False)
    tf.config.experimental.enable_op_determinism()
    rows, predictions, models = [], [], []
    for regime, (scaler, splits) in prepared.items():
        for seed in p.seeds:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_model(p.lookback)
            train, val = splits['train'], splits['validation']
            history = model.fit(dataset(*train[:2]), validation_data=dataset(*val[:2]),
                                epochs=p.epochs, shuffle=False, verbose=0,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=p.patience, restore_best_weights=True)])
            name = f'{regime}_seed{seed}'
            model.save(out/f'{name}.keras')
            pd.DataFrame(history.history).to_csv(out/f'{name}_history.csv', index=False)
            (out/f'{name}_scaler.json').write_text(json.dumps({
                'min': scaler.min_.tolist(), 'scale': scaler.scale_.tolist(),
                'data_min': scaler.data_min_.tolist(), 'data_max': scaler.data_max_.tolist(),
                'training_samples': len(train[2]), 'best_epoch': int(np.argmin(history.history['val_loss'])+1),
                'train_target_dates': df.Date.iloc[train[2]].dt.strftime('%Y-%m-%d').tolist()}, indent=2))
            idx = val[2]
            pred = scaler.inverse_transform(model(val[0], training=False).numpy()).ravel()
            actual, previous = df.Close.to_numpy()[idx], df.Close.to_numpy()[idx-1]
            rows.append({'Split': 'validation', 'Model': regime, 'Seed': seed, **evaluate(actual,pred,previous)})
            predictions.append(pd.DataFrame({'Date':df.Date.iloc[idx].to_numpy(), 'Split':'validation', 'Model':regime,
                                             'Seed':seed,'Actual':actual,'Prediction':pred,'PreviousClose':previous}))
            models.append((regime, seed, name))
    validation = pd.DataFrame(rows)
    winner = validation.groupby('Model').MSE.mean().sort_values().index[0]
    # Persist the selection BEFORE accessing test predictions, and never tune after it.
    (out/'selection.json').write_text(json.dumps({'selected_regime':winner,
        'mean_validation_mse':validation.groupby('Model').MSE.mean().to_dict(),
        'test_used_for_selection':False},indent=2))
    for regime, seed, name in models:
        scaler, splits = prepared[regime]
        x, _, idx = splits['test']
        model = tf.keras.models.load_model(out/f'{name}.keras')
        pred = scaler.inverse_transform(model(x,training=False).numpy()).ravel()
        actual, previous = df.Close.to_numpy()[idx], df.Close.to_numpy()[idx-1]
        rows.append({'Split':'test','Model':regime,'Seed':seed,**evaluate(actual,pred,previous)})
        predictions.append(pd.DataFrame({'Date':df.Date.iloc[idx].to_numpy(),'Split':'test','Model':regime,
                                         'Seed':seed,'Actual':actual,'Prediction':pred,'PreviousClose':previous}))
    common = next(iter(prepared.values()))[1]
    for split in ('validation','test'):
        idx = common[split][2]
        actual, previous = df.Close.to_numpy()[idx], df.Close.to_numpy()[idx-1]
        rows.append({'Split':split,'Model':'Naive','Seed':None,**evaluate(actual,previous,previous)})
        predictions.append(pd.DataFrame({'Date':df.Date.iloc[idx].to_numpy(),'Split':split,'Model':'Naive',
                                         'Seed':None,'Actual':actual,'Prediction':previous,'PreviousClose':previous}))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    keys = ['MSE','RMSE','MAE','R2','DirectionalAccuracy']
    summary = metrics.groupby(['Split','Model'])[keys].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    print('Selected using validation only:',winner)
    print(summary.to_string())
    return metrics, winner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    run_experiment(args.csv,args.output)
