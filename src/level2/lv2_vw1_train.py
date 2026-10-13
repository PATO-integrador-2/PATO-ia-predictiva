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


PRIVATION_COLS = [
    'personas_que_habitan_ese_hogar',
    'privacion_bajo_logro_educativo',
    'privacion_analfabetismo',
    'privacion_inasistencia_escolar',
    'privacion_rezago_escolar',
    'privacion_atencion_integral_primera_infancia',
    'privacion_trabajo_infantil',
    'privacion_no_aseguramiento_salud',
    'privacion_barreras_acceso_salud',
    'privacion_desempleo_larga_duracion',
    'privacion_tasa_empleo_formal',
    'privacion_no_acceso_agua_mejorada',
    'privacion_inadecuada_eliminacion_excretas',
    'privacion_inadecuado_material_pisos',
    'privacion_inadecuado_material_paredes',
    'privacion_hacinamiento_critico',
]
FEATURE_COLS = [
    'year', 'year_squared', 'years_since_2010',
    'ipm_lag1', 'ipm_lag2', 'ipm_lag3', 'ipm_trend',
    *PRIVATION_COLS, 'region_encoded', 'departamento_encoded'
]
GROUP_COLS = ['region', 'departamento']


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
    frame['region'] = frame['region'].astype(str)
    frame['departamento'] = frame['departamento'].astype(str)
    frame['anio'] = pd.to_numeric(frame['anio'], errors='raise').astype(int)
    frame['ipm'] = pd.to_numeric(frame['ipm'], errors='raise')
    # La vista trae un registro por hogar: se promedia por departamento y
    # año (igual que DepartmentPovertyPredictor._aggregate_by_year) en vez
    # de quedarse con un hogar arbitrario por grupo.
    frame = (
        frame.groupby(GROUP_COLS + ['anio'], as_index=False)[[*PRIVATION_COLS, 'ipm']]
        .mean()
        .sort_values(GROUP_COLS + ['anio'])
    )
    frame['year'] = frame['anio']
    frame['year_squared'] = frame['anio'] ** 2
    frame['years_since_2010'] = frame['anio'] - 2010

    region_encoder = LabelEncoder()
    department_encoder = LabelEncoder()
    frame['region_encoded'] = region_encoder.fit_transform(frame['region'])
    frame['departamento_encoded'] = department_encoder.fit_transform(frame['departamento'])
    grouped = frame.groupby(GROUP_COLS)['ipm']
    frame['ipm_lag1'] = grouped.shift(1)
    frame['ipm_lag2'] = grouped.shift(2)
    frame['ipm_lag3'] = grouped.shift(3)
    changes = grouped.diff()
    frame['ipm_trend'] = changes.groupby([frame['region'], frame['departamento']]).transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    frame = frame.dropna(subset=FEATURE_COLS + ['ipm'])

    available_years = sorted(frame['anio'].unique().tolist())
    train_years = [year for year in available_years if 2018 <= year <= 2022]
    test_years = [year for year in available_years if 2023 <= year <= 2025]
    if len(train_years) < 2 or not test_years:
        split_index = max(1, int(len(available_years) * 0.8))
        train_years = available_years[:split_index]
        test_years = available_years[split_index:]
    train_mask = frame['anio'].isin(train_years)
    test_mask = frame['anio'].isin(test_years)
    if not train_mask.any() or not test_mask.any():
        raise ValueError('Nivel 2 no tiene años suficientes para un split temporal')

    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame.loc[train_mask, FEATURE_COLS])
    x_test = scaler.transform(frame.loc[test_mask, FEATURE_COLS])
    return {
        'frame': frame,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': frame.loc[train_mask, 'ipm'].to_numpy(),
        'y_test': frame.loc[test_mask, 'ipm'].to_numpy(),
        'test_lag1': frame.loc[test_mask, 'ipm_lag1'].to_numpy(),
        'scaler': scaler,
        'region_encoder': region_encoder,
        'department_encoder': department_encoder,
        'train_years': train_years,
        'test_years': test_years,
    }


def train_lv2_vw1():
    data = _prepare_data(Database().get_data_view_household_poverty_by_department())
    candidates = {'naive': data['test_lag1']}
    fitted_models = {}
    unavailable_models = []
    tree_models = {
        'random_forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=4,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    }
    try:
        from xgboost import XGBRegressor
        tree_models['xgboost'] = XGBRegressor(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            objective='reg:squarederror', random_state=42, n_jobs=-1
        )
    except ImportError:
        unavailable_models.append('xgboost')
    try:
        from lightgbm import LGBMRegressor
        tree_models['lightgbm'] = LGBMRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            random_state=42, verbosity=-1, n_jobs=-1
        )
    except ImportError:
        unavailable_models.append('lightgbm')

    for name, model in tree_models.items():
        model.fit(data['x_train'], data['y_train'])
        fitted_models[name] = model
        candidates[name] = model.predict(data['x_test'])

    candidate_metrics = {name: _metrics(data['y_test'], prediction) for name, prediction in candidates.items()}
    selected_model, variation_required = select_best_model(candidates, candidate_metrics)
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'models', 'train_lv2_vw1'
    )
    os.makedirs(model_dir, exist_ok=True)
    selected_artifact = fitted_models.get(selected_model, fitted_models.get('random_forest'))
    joblib.dump(selected_artifact, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(fitted_models, os.path.join(model_dir, 'candidate_models.pkl'))
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(data['region_encoder'], os.path.join(model_dir, 'label_encoder_region.pkl'))
    joblib.dump(data['department_encoder'], os.path.join(model_dir, 'label_encoder_departamento.pkl'))
    joblib.dump(FEATURE_COLS, os.path.join(model_dir, 'feature_cols.pkl'))

    metrics = {
        'model': 'temporal_benchmark',
        'selected_model': selected_model,
        'variation_required': variation_required,
        'candidates': candidate_metrics,
        'available_models': sorted(candidates),
        'unavailable_models': unavailable_models,
        'feature_names': FEATURE_COLS,
        'train_years': [int(year) for year in data['train_years']],
        'test_years': [int(year) for year in data['test_years']],
        'regions': sorted(data['frame']['region'].unique().tolist()),
        'departments': sorted(data['frame']['departamento'].unique().tolist()),
        'n_samples': int(len(data['frame'])),
        'train_date': datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    return selected_artifact, metrics


if __name__ == '__main__':
    train_lv2_vw1()