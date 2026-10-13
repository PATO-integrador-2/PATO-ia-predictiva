import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database


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


def linear(x, a, b):
    return a + b * x


def quadratic(x, a, b, c):
    return a + b * x + c * x ** 2


def logarithmic(x, a, b):
    return a + b * np.log(x + 1)


def exponential(x, a, b):
    return a * np.exp(b * x)


def project_privation(historical_values, future_years):
    values = np.asarray(historical_values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return np.repeat(np.clip(values[-1], 0, 1), future_years)
    x_data = np.arange(len(values))
    candidates = (linear, quadratic, logarithmic, exponential)
    best_function = None
    best_params = None
    best_r2 = -float('inf')
    for function in candidates:
        try:
            params, _ = curve_fit(function, x_data, values, maxfev=5000)
            score = r2_score(values, function(x_data, *params))
            if score > best_r2:
                best_function, best_params, best_r2 = function, params, score
        except Exception:
            continue
    if best_function is None:
        return np.repeat(np.clip(values[-1], 0, 1), future_years)
    future = best_function(np.arange(len(values), len(values) + future_years), *best_params)
    return np.clip(future, 0, 1)


class DepartmentPovertyPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(base_dir, 'models', 'train_lv2_vw1')
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        try:
            self.candidate_models = joblib.load(os.path.join(self.model_dir, 'candidate_models.pkl'))
        except FileNotFoundError:
            self.candidate_models = {}
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_region = joblib.load(os.path.join(self.model_dir, 'label_encoder_region.pkl'))
        self.le_departamento = joblib.load(os.path.join(self.model_dir, 'label_encoder_departamento.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        with open(os.path.join(self.model_dir, 'metrics.json'), encoding='utf-8') as file:
            self.metrics = json.load(file)
        self.selected_model = self.metrics.get('selected_model', 'random_forest')

    def _aggregate_by_year(self, data_frame):
        columns = [*PRIVATION_COLS, 'ipm']
        return data_frame.groupby('anio')[columns].mean().reset_index()

    def _get_historical_data(self, region, departamento, years_hist=5):
        frame = Database().get_data_view_household_poverty_by_department().copy()
        frame['region'] = frame['region'].astype(str)
        frame['departamento'] = frame['departamento'].astype(str)
        frame = frame[(frame['region'] == str(region)) & (frame['departamento'] == str(departamento))]
        frame = self._aggregate_by_year(frame).sort_values('anio')
        if len(frame) < years_hist:
            raise ValueError(f'Datos insuficientes para {region} - {departamento}')
        recent = frame.tail(years_hist)
        years = recent['anio'].to_numpy()
        if not np.all(np.diff(years) == 1):
            raise ValueError(f'La ventana reciente contiene años faltantes: {years.tolist()}')
        if not np.all(np.diff(frame['anio'].to_numpy()) == 1):
            print(f'Advertencia: hay huecos históricos; se usará la ventana {years.tolist()}')
        return recent

    def _prepare_features(self, row, model=None):
        values = {}
        for column in self.feature_cols:
            if column == 'region_encoded':
                values[column] = self.le_region.transform([str(row['region'])])[0]
            elif column == 'departamento_encoded':
                values[column] = self.le_departamento.transform([str(row['departamento'])])[0]
            else:
                values[column] = row[column]
        feature_frame = pd.DataFrame([[values[column] for column in self.feature_cols]], columns=self.feature_cols)
        return self.scaler.transform(feature_frame), (model or self.model)

    def predict_household(self, row):
        features_scaled, model = self._prepare_features(row, self.candidate_models.get(self.selected_model))
        return float(np.clip(model.predict(features_scaled)[0], 0, 1))

    def predict_department_trend(self, region, departamentos, año_inicio=2026, años=5):
        results = []
        for departamento in departamentos:
            # Los errores (datos insuficientes, base caída) se propagan: un
            # departamento sin datos no debe devolver éxito con lista vacía.
            historical = self._get_historical_data(region, departamento)
            ipm_values = historical['ipm'].astype(float).tolist()
            projections = {
                column: (
                    [historical[column].mean()] * años
                    if column == 'personas_que_habitan_ese_hogar'
                    else project_privation(historical[column].values, años)
                )
                for column in PRIVATION_COLS
            }
            ipm_real = ipm_values[-1]
            real_year = int(historical['anio'].iloc[-1])
            for offset in range(años):
                year = año_inicio + offset
                row = {'region': region, 'departamento': departamento}
                row.update({
                    'year': year,
                    'year_squared': year ** 2,
                    'years_since_2010': year - 2010,
                    'ipm_lag1': ipm_values[-1],
                    'ipm_lag2': ipm_values[-2],
                    'ipm_lag3': ipm_values[-3],
                    'ipm_trend': float(np.mean(np.diff(ipm_values)[-3:])),
                })
                row.update({column: projections[column][offset] for column in PRIVATION_COLS})
                prediction = self.predict_household(row)
                results.append({
                    'departamento': departamento,
                    'region': region,
                    'anio_real': real_year,
                    'ipm_real': float(ipm_real),
                    'año': year,
                    'ipm_predicho': prediction,
                    'cambio': prediction - ipm_real,
                    'cambio_porcentaje': ((prediction - ipm_real) / ipm_real * 100) if ipm_real else 0,
                })
                ipm_values.append(prediction)
        return pd.DataFrame(results)

    def predict_household_poverty(self, region, departamento, household_data):
        row = {'region': region, 'departamento': departamento, **household_data}
        row.setdefault('year', 2025)
        row.setdefault('year_squared', row['year'] ** 2)
        row.setdefault('years_since_2010', row['year'] - 2010)
        row.setdefault('ipm_lag1', 0)
        row.setdefault('ipm_lag2', 0)
        row.setdefault('ipm_lag3', 0)
        row.setdefault('ipm_trend', 0)
        return self.predict_household(row)

    def predict_household_trend(self, region, departamento, household_data, año_inicio=2026, años=5):
        # Los rezagos de IPM salen del historial del departamento (igual que
        # predict_department_trend); con rezagos en 0 el modelo subestima el IPM.
        historical = self._get_historical_data(region, departamento)
        ipm_values = historical['ipm'].astype(float).tolist()
        predictions = []
        for offset in range(años):
            year = año_inicio + offset
            row = {**household_data, 'region': region, 'departamento': departamento}
            row.update({
                'year': year,
                'year_squared': year ** 2,
                'years_since_2010': year - 2010,
                'ipm_lag1': ipm_values[-1],
                'ipm_lag2': ipm_values[-2],
                'ipm_lag3': ipm_values[-3],
                'ipm_trend': float(np.mean(np.diff(ipm_values)[-3:])),
            })
            prediction = self.predict_household(row)
            predictions.append({'year': year, 'predicted_ipm': round(prediction, 4)})
            ipm_values.append(prediction)
        return predictions


if __name__ == '__main__':
    predictor = DepartmentPovertyPredictor()
    print(predictor.predict_department_trend('Caribe', ['Atlántico'], 2026, 5).to_string(index=False))