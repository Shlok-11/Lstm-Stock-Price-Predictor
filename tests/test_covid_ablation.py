from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from experiments.covid_ablation import Protocol, prepare_experiment, run_experiment
from src.stock_prediction import load_prices


def history():
    dates = pd.bdate_range('2015-01-01', '2024-12-31')
    return pd.DataFrame({'Date':dates,'Close':100+np.arange(len(dates))*.01})


def test_no_splice_no_2020_and_common_holdouts():
    df = history()
    # Excluded regime's scaler must ignore COVID outliers and future values.
    df.loc[df.Date.dt.year == 2020,'Close'] = 10000
    df.loc[df.Date >= '2023-01-01','Close'] = 20000
    source, prepared, skipped = prepare_experiment(df)
    assert not skipped
    original = prepared['with_2020'][1]
    assert prepared['with_2020'][0].data_max_[0] == 10000
    assert prepared['exclude_2020'][0].data_max_[0] < 10000
    for name,(scaler,splits) in prepared.items():
        assert scaler.data_max_[0] < 20000
        for split in ('validation','test'):
            np.testing.assert_array_equal(splits[split][2],original[split][2])
            np.testing.assert_allclose(scaler.inverse_transform(splits[split][1]).ravel(), source.Close.iloc[splits[split][2]],rtol=1e-6)
        for i in splits['train'][2]:
            dates=source.Date.iloc[i-20:i+1]
            assert len(dates)==21 and dates.max() < pd.Timestamp('2023-01-01')
            if name=='exclude_2020': assert not (dates.dt.year==2020).any()
            if name=='post_2020': assert dates.min() >= pd.Timestamp('2021-01-01')
    # First 20 post-gap rows are ineligible, even though their target is not 2020.
    first = source.index[source.Date >= '2021-01-01'][0]
    idx = prepared['exclude_2020'][1]['train'][2]
    assert not np.isin(np.arange(first,first+20),idx).any()
    assert first+20 in idx


def test_historical_dataset_cannot_answer_covid_question():
    with pytest.raises(ValueError,match='pre-2020 and 2020'):
        prepare_experiment(load_prices('data/aapl_history.csv'))


def test_source_outage_rejected():
    df=history()
    df=df.loc[~df.Date.between('2020-01-01','2020-02-01')].reset_index(drop=True)
    with pytest.raises(ValueError,match='Source gap'):prepare_experiment(df)


def test_post_period_optional_when_insufficient():
    p=replace(Protocol(),post_start='2022-12-01')
    _,prepared,skipped=prepare_experiment(history(),p)
    assert 'post_2020' in skipped and 'post_2020' not in prepared


def test_missing_test_rejected():
    with pytest.raises(ValueError,match='test observations'):
        prepare_experiment(history().query("Date < '2024-01-01'"))


def test_three_regime_training_export_and_selection(tmp_path):
    df=history(); csv=tmp_path/'input.csv'; df.to_csv(csv,index=False)
    out=tmp_path/'new_run'
    p=replace(Protocol(),epochs=1,seeds=(42,))
    metrics,winner=run_experiment(csv,out,p)
    assert len(metrics)==8
    selection=json.loads((out/'selection.json').read_text())
    assert not selection['test_used_for_selection']
    scores=metrics.query("Split=='validation' and Model!='Naive'").set_index('Model').MSE
    assert winner == scores.idxmin()
    predictions=pd.read_csv(out/'predictions.csv')
    for split in ['validation','test']:
        group=predictions.query('Split==@split')
        dates=[g.Date.tolist() for _,g in group.groupby('Model')]
        assert all(d==dates[0] for d in dates)
        naive=group.query("Model=='Naive'")
        np.testing.assert_array_equal(naive.Prediction,naive.PreviousClose)
    assert np.isfinite(metrics[['MSE','RMSE','MAE','R2','DirectionalAccuracy']]).all().all()
    with pytest.raises(FileExistsError):run_experiment(csv,out,p)
