import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database
from model_selection import select_best_model


FEATURE_COLS = [
    'year', 'years_since_2010', 'lag1', 'lag2', 'lag3',
    'ma2', 'ma3', 'trend', 'domain_encoded'
]


def _mape(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = np.abs(actual) > 1e-8
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if np.any(mask) else None


def _metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mape': _mape(y_true, y_pred),
        'r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


def _prepare_data(data_frame):
    frame = data_frame.dropna().copy()
    frame = frame[~frame['dominio'].str.contains('__test__', na=False)]
    frame['dominio'] = frame['dominio'].astype(str)
    frame['anio'] = pd.to_numeric(frame['anio'], errors='raise').astype(int)
    frame['porcentaje'] = pd.to_numeric(frame['porcentaje'], errors='raise')
    frame = frame.drop_duplicates(subset=['dominio', 'anio']).sort_values(['dominio', 'anio'])
    frame['year'] = frame['anio']
    frame['years_since_2010'] = frame['anio'] - 2010

    grouped = frame.groupby('dominio')['porcentaje']
    frame['lag1'] = grouped.shift(1)
    frame['lag2'] = grouped.shift(2)
    frame['lag3'] = grouped.shift(3)
    frame['ma2'] = grouped.transform(lambda values: values.shift(1).rolling(2, min_periods=2).mean())
    frame['ma3'] = grouped.transform(lambda values: values.shift(1).rolling(3, min_periods=3).mean())
    changes = grouped.diff()
    frame['trend'] = changes.groupby(frame['dominio']).transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )

    label_encoder = LabelEncoder()
    frame['domain_encoded'] = label_encoder.fit_transform(frame['dominio'])
    frame = frame.dropna(subset=FEATURE_COLS + ['porcentaje'])

    available_years = sorted(frame['year'].unique().tolist())
    train_years = [year for year in available_years if 2018 <= year <= 2022]
    test_years = [year for year in available_years if 2023 <= year <= 2025]
    if len(train_years) < 2 or not test_years:
        split_index = max(1, int(len(available_years) * 0.8))
        train_years = available_years[:split_index]
        test_years = available_years[split_index:]

    train_mask = frame['year'].isin(train_years)
    test_mask = frame['year'].isin(test_years)
    if not train_mask.any() or not test_mask.any():
        raise ValueError('La vista 3 no tiene años suficientes para un split temporal')

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_mask, FEATURE_COLS])
    x_test = scaler.transform(frame.loc[test_mask, FEATURE_COLS])
    return {
        'frame': frame,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': frame.loc[train_mask, 'porcentaje'].to_numpy(),
        'y_test': frame.loc[test_mask, 'porcentaje'].to_numpy(),
        'test_lag1': frame.loc[test_mask, 'lag1'].to_numpy(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'train_years': train_years,
        'test_years': test_years,
    }


def _fit_arima_predictions(frame, train_years, test_years):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return None, None

    predictions = []
    fitted_models = {}
    for domain, series in frame.groupby('dominio'):
        train_series = series[series['year'].isin(train_years)].sort_values('year')
        test_series = series[series['year'].isin(test_years)].sort_values('year')
        if len(train_series) < 3:
            continue
        try:
            model = ARIMA(train_series['porcentaje'].to_numpy(), order=(1, 1, 0)).fit()
            forecast = model.forecast(steps=len(test_series))
            predictions.extend((domain, year, value) for year, value in zip(test_series['year'], forecast))
            fitted_models[domain] = ARIMA(series['porcentaje'].to_numpy(), order=(1, 1, 0)).fit()
        except Exception:
            continue

    if not predictions:
        return None, None
    prediction_frame = pd.DataFrame(predictions, columns=['dominio', 'year', 'prediction'])
    test_frame = frame[frame['year'].isin(test_years)][['dominio', 'year', 'porcentaje']]
    merged = test_frame.merge(prediction_frame, on=['dominio', 'year'], how='inner')
    return merged['prediction'].to_numpy(), fitted_models


def train_lv1_vw3():
    data = _prepare_data(Database().get_data_view_average_deprivations())
    candidates = {'naive': data['test_lag1']}
    fitted_models = {}
    unavailable_models = []
    tree_models = {
        'random_forest': RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )
    }
    try:
        from xgboost import XGBRegressor
        tree_models['xgboost'] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective='reg:squarederror', random_state=42, n_jobs=-1
        )
    except ImportError:
        unavailable_models.append('xgboost')
    try:
        from lightgbm import LGBMRegressor
        tree_models['lightgbm'] = LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, verbosity=-1, n_jobs=-1
        )
    except ImportError:
        unavailable_models.append('lightgbm')

    for name, model in tree_models.items():
        model.fit(data['x_train'], data['y_train'])
        fitted_models[name] = model
        candidates[name] = model.predict(data['x_test'])

    arima_prediction, arima_models = _fit_arima_predictions(
        data['frame'], data['train_years'], data['test_years']
    )
    if arima_prediction is not None and len(arima_prediction) == len(data['y_test']):
        candidates['arima'] = arima_prediction
    else:
        unavailable_models.append('arima')

    candidate_metrics = {name: _metrics(data['y_test'], prediction) for name, prediction in candidates.items()}
    selected_model, variation_required = select_best_model(candidates, candidate_metrics)
    test_meta = data['frame'].loc[data['frame']['year'].isin(data['test_years']), ['dominio']].reset_index(drop=True)
    metrics_by_domain = {}
    selected_model_by_domain = {}
    for domain in sorted(test_meta['dominio'].unique()):
        domain_mask = test_meta['dominio'].eq(domain).to_numpy()
        domain_metrics = {
            name: _metrics(data['y_test'][domain_mask], prediction[domain_mask])
            for name, prediction in candidates.items()
        }
        metrics_by_domain[domain] = domain_metrics
        selected_model_by_domain[domain], _ = select_best_model(
            {name: prediction[domain_mask] for name, prediction in candidates.items()},
            domain_metrics
        )

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'models', 'train_lv1_vw3'
    )
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(fitted_models.get(selected_model, fitted_models.get('random_forest')), os.path.join(model_dir, 'model.pkl'))
    joblib.dump(fitted_models, os.path.join(model_dir, 'candidate_models.pkl'))
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(data['label_encoder'], os.path.join(model_dir, 'label_encoder_domain.pkl'))
    joblib.dump(FEATURE_COLS, os.path.join(model_dir, 'feature_cols.pkl'))
    joblib.dump(arima_models or {}, os.path.join(model_dir, 'arima_models.pkl'))

    metrics = {
        'model': 'temporal_benchmark',
        'selected_model': selected_model,
        'variation_required': variation_required,
        'selected_model_by_domain': selected_model_by_domain,
        'candidates': candidate_metrics,
        'metrics_by_domain': metrics_by_domain,
        'available_models': sorted(candidates),
        'unavailable_models': unavailable_models,
        'feature_names': FEATURE_COLS,
        'train_years': [int(year) for year in data['train_years']],
        'test_years': [int(year) for year in data['test_years']],
        'domains': sorted(data['frame']['dominio'].unique().tolist()),
        'n_samples': int(len(data['frame'])),
        'train_date': datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    return fitted_models.get(selected_model), metrics


if __name__ == '__main__':
    train_lv1_vw3()