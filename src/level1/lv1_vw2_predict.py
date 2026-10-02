import json
import os
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database


class PrivationsPredictor:
    """Pronostica una privación por dominio usando el modelo seleccionado temporalmente."""

    def __init__(self, usar_global: bool = True):
        self.usar_global = usar_global
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(base_dir, 'models', 'train_lv1_vw2')
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        try:
            self.candidate_models = joblib.load(os.path.join(self.model_dir, 'candidate_models.pkl'))
        except FileNotFoundError:
            self.candidate_models = {}
        self.scaler_global = joblib.load(os.path.join(self.model_dir, 'scaler_global.pkl'))
        self.le_domain = joblib.load(os.path.join(self.model_dir, 'label_encoder_domain.pkl'))
        self.le_variable = joblib.load(os.path.join(self.model_dir, 'label_encoder_variable.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        try:
            self.arima_models = joblib.load(os.path.join(self.model_dir, 'arima_models.pkl'))
        except FileNotFoundError:
            self.arima_models = {}
        try:
            with open(os.path.join(self.model_dir, 'metrics.json'), encoding='utf-8') as file:
                self.metrics = json.load(file)
        except FileNotFoundError:
            self.metrics = {}

    def _resolve_variable(self, variable: str) -> str:
        """Accepts the catalog name with or without the API display prefix."""
        accepted = list(self.le_variable.classes_)
        normalized = self._normalize_text(variable)
        candidates = [normalized]
        for prefix in ('privacion por ', 'privación por '):
            if normalized.startswith(prefix):
                candidates.append(normalized[len(prefix):])

        for option in accepted:
            option_normalized = self._normalize_text(option)
            if option_normalized in candidates:
                return option
        return variable

    @staticmethod
    def _normalize_text(value: str) -> str:
        text = unicodedata.normalize('NFKD', str(value).strip().lower())
        return ''.join(character for character in text if not unicodedata.combining(character))

    def _get_historical_data(self, variable: str, domain: str):
        frame = Database().get_data_view_deprivations_variable()
        frame = frame[(frame['variable'] == variable) & (frame['dominio'] == domain)].copy()
        frame['anio'] = pd.to_numeric(frame['anio'], errors='raise').astype(int)
        frame['ipm'] = pd.to_numeric(frame['ipm'], errors='raise')
        frame = frame.drop_duplicates(subset=['dominio', 'variable', 'anio']).sort_values('anio')
        if len(frame) < 4:
            raise ValueError(f'Datos insuficientes para {variable} - {domain}')

        recent = frame.tail(4)
        years = recent['anio'].to_numpy()
        if not np.all(np.diff(years) == 1):
            raise ValueError(f'La ventana reciente contiene años faltantes: {years.tolist()}')
        if not np.all(np.diff(frame['anio'].to_numpy()) == 1):
            print(f'Advertencia: hay huecos históricos; se usará la ventana {years.tolist()}')
        return recent

    @staticmethod
    def _trend(values):
        changes = np.diff(values)
        return float(np.mean(changes[-3:])) if len(changes) else 0.0

    def _predict_tree(self, model, values, year, domain_encoded, variable_encoded):
        features = {
            'year': year,
            'year_squared': year ** 2,
            'years_since_2010': year - 2010,
            'ipm_lag1': values[-1],
            'ipm_lag2': values[-2],
            'ipm_lag3': values[-3],
            'domain_trend': self._trend(values),
            'variable_trend': self._trend(values),
            'domain_encoded': domain_encoded,
            'variable_encoded': variable_encoded,
        }
        feature_frame = pd.DataFrame(
            [[features[column] for column in self.feature_cols]],
            columns=self.feature_cols
        )
        scaled_features = self.scaler_global.transform(feature_frame)
        return float(model.predict(scaled_features)[0])

    def predict(self, variable: str, domain: str, begin_year: int = 2026, years: int = 5):
        variable = self._resolve_variable(variable)
        historical = self._get_historical_data(variable, domain)
        values = historical['ipm'].astype(float).tolist()
        domain_encoded = self.le_domain.transform([domain])[0]
        variable_encoded = self.le_variable.transform([variable])[0]
        selected_model = self.metrics.get('selected_model', 'random_forest')
        selected_model = self.metrics.get('selected_model_by_variable', {}).get(
            variable, selected_model
        )
        arima_model = self.arima_models.get((domain, variable))
        predictions = []

        for offset in range(years):
            year = begin_year + offset
            if selected_model == 'naive':
                prediction = values[-1]
            elif selected_model == 'arima' and arima_model is not None:
                prediction = float(arima_model.forecast(steps=offset + 1)[-1])
            else:
                model = self.candidate_models.get(selected_model, self.model)
                prediction = self._predict_tree(
                    model, values[-4:], year, domain_encoded, variable_encoded
                )

            prediction = max(float(prediction), 0.0)
            predictions.append({
                'year': year,
                'variable': variable,
                'domain': domain,
                'ipm_foretold': prediction,
            })
            values.append(prediction)

        return pd.DataFrame(predictions)


def predict_privacion(variable: str, domain: str, begin_year: int = 2026, years: int = 5):
    return PrivationsPredictor().predict(variable, domain, begin_year, years)


if __name__ == '__main__':
    predictor = PrivationsPredictor()
    result = predictor.predict('Analfabetismo', 'Nacional', 2026, 5)
    print(result.to_string(index=False))