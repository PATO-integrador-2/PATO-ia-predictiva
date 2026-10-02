import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database


class PovertyByAgePredictorV2:
    """Pronostica pobreza por país, área y rango de edad."""

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(base_dir, 'models', 'train_lv3_vw3')
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        try:
            self.candidate_models = joblib.load(os.path.join(self.model_dir, 'candidate_models.pkl'))
        except FileNotFoundError:
            self.candidate_models = {}
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_area = joblib.load(os.path.join(self.model_dir, 'label_encoder_area.pkl'))
        self.le_country = joblib.load(os.path.join(self.model_dir, 'label_encoder_country.pkl'))
        self.le_age_range = joblib.load(os.path.join(self.model_dir, 'label_encoder_age_range.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.areas = joblib.load(os.path.join(self.model_dir, 'areas.pkl'))
        self.countries = joblib.load(os.path.join(self.model_dir, 'countries.pkl'))
        self.age_ranges = joblib.load(os.path.join(self.model_dir, 'age_ranges.pkl'))
        with open(os.path.join(self.model_dir, 'metrics.json'), encoding='utf-8') as file:
            self.metrics = json.load(file)

    def _get_historical_data(self, country: str, area: str, age_range: str):
        frame = Database().get_data_view_dashboard03_poverty_by_age()
        frame = frame[
            (frame['pais'] == country) &
            (frame['area_geografica'] == area) &
            (frame['grupo_erario'] == age_range)
        ].copy()
        frame['anio'] = pd.to_numeric(frame['anio'], errors='raise').astype(int)
        frame['valor_porcentaje'] = pd.to_numeric(frame['valor_porcentaje'], errors='raise')
        frame = frame.drop_duplicates(subset=['pais', 'area_geografica', 'grupo_erario', 'anio']).sort_values('anio')
        if len(frame) < 5:
            raise ValueError(f'Datos insuficientes para {country} - {area} - {age_range}')
        recent = frame.tail(5)
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

    def _predict_tree(self, model, values, year, area_encoded, country_encoded, age_encoded):
        feature_values = {
            'year': year, 'year_squared': year ** 2,
            'years_since_2010': year - 2010,
            'value_lag1': values[-1], 'value_lag2': values[-2],
            'avg_previous_2': float(np.mean(values[-2:])),
            'avg_previous_3': float(np.mean(values[-3:])),
            'value_trend': self._trend(values),
            'area_encoded': area_encoded, 'country_encoded': country_encoded,
            'age_range_encoded': age_encoded,
        }
        feature_frame = pd.DataFrame(
            [[feature_values[column] for column in self.feature_cols]],
            columns=self.feature_cols
        )
        return float(model.predict(self.scaler.transform(feature_frame))[0])

    def predict(self, country: str, area: str, age_range: str, begin_year: int = 2026, years: int = 5):
        historical = self._get_historical_data(country, area, age_range)
        values = historical['valor_porcentaje'].astype(float).tolist()
        area_encoded = self.le_area.transform([area])[0]
        country_encoded = self.le_country.transform([country])[0]
        age_encoded = self.le_age_range.transform([age_range])[0]
        series_key = f'{country}::{area}::{age_range}'
        selected_model = self.metrics.get('selected_model_by_series', {}).get(
            series_key, self.metrics.get('selected_model', 'random_forest')
        )
        predictions = []
        for offset in range(years):
            year = begin_year + offset
            if selected_model == 'naive':
                prediction = values[-1]
            else:
                model = self.candidate_models.get(selected_model, self.model)
                prediction = self._predict_tree(model, values[-3:], year, area_encoded, country_encoded, age_encoded)
            prediction = max(0.0, min(float(prediction), 100.0))
            predictions.append({
                'year': year, 'country': country, 'area': area,
                'age_range': age_range, 'predicted_value': prediction,
            })
            values.append(prediction)
        return pd.DataFrame(predictions)

    def predict_comparative_age_ranges(self, country: str, area: str, age_ranges: list, begin_year: int = 2026, years: int = 5):
        results = {age_range: self.predict(country, area, age_range, begin_year, years) for age_range in age_ranges}
        comparison = pd.DataFrame({'Year': list(range(begin_year, begin_year + years))})
        for age_range, frame in results.items():
            comparison[age_range] = frame['predicted_value'].values
        return comparison


def predict_poverty_by_age_v2(country: str, area: str, age_range: str, begin_year: int = 2026, years: int = 5):
    return PovertyByAgePredictorV2().predict(country, area, age_range, begin_year, years)


def predict_comparative_age_ranges_v2(country: str, area: str, age_ranges: list, begin_year: int = 2026, years: int = 5):
    return PovertyByAgePredictorV2().predict_comparative_age_ranges(country, area, age_ranges, begin_year, years)


if __name__ == '__main__':
    print(PovertyByAgePredictorV2().predict('Latin America (weighted average)', 'National', '0 a 17 años', 2026, 5).to_string(index=False))