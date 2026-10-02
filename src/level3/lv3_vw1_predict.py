import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database


class ContributionLatamPredictor:
    """Pronostica contribuciones por país y privación."""

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(base_dir, 'models', 'train_lv3_vw1')
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        try:
            self.candidate_models = joblib.load(os.path.join(self.model_dir, 'candidate_models.pkl'))
        except FileNotFoundError:
            self.candidate_models = {}
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_country = joblib.load(os.path.join(self.model_dir, 'label_encoder_country.pkl'))
        self.le_privation = joblib.load(os.path.join(self.model_dir, 'label_encoder_privation.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.countries = joblib.load(os.path.join(self.model_dir, 'countries.pkl'))
        self.privations = joblib.load(os.path.join(self.model_dir, 'privations.pkl'))
        with open(os.path.join(self.model_dir, 'metrics.json'), encoding='utf-8') as file:
            self.metrics = json.load(file)

    def _get_historical_data(self, privation: str, country: str):
        frame = Database().get_data_view_dashboard03_deprivation_contribution()
        frame = frame[(frame['privacion'] == privation) & (frame['pais'] == country)].copy()
        frame['anio'] = pd.to_numeric(frame['anio'], errors='raise').astype(int)
        frame['valor_porcentaje'] = pd.to_numeric(frame['valor_porcentaje'], errors='raise')
        frame = frame.drop_duplicates(subset=['pais', 'privacion', 'anio']).sort_values('anio')
        if len(frame) < 4:
            raise ValueError(f'Datos insuficientes para {privation} - {country}')
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

    def _predict_tree(self, model, values, year, country_encoded, privation_encoded):
        feature_values = {
            'year': year,
            'year_squared': year ** 2,
            'years_since_2010': year - 2010,
            'value_lag1': values[-1],
            'value_lag2': values[-2],
            'value_lag3': values[-3],
            'value_trend': self._trend(values),
            'country_encoded': country_encoded,
            'privation_encoded': privation_encoded,
        }
        feature_frame = pd.DataFrame(
            [[feature_values[column] for column in self.feature_cols]],
            columns=self.feature_cols
        )
        return float(model.predict(self.scaler.transform(feature_frame))[0])

    def predict(self, privation: str, country: str, begin_year: int = 2026, years: int = 5):
        historical = self._get_historical_data(privation, country)
        values = historical['valor_porcentaje'].astype(float).tolist()
        country_encoded = self.le_country.transform([country])[0]
        privation_encoded = self.le_privation.transform([privation])[0]
        series_key = f'{country}::{privation}'
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
                prediction = self._predict_tree(
                    model, values[-3:], year, country_encoded, privation_encoded
                )
            prediction = max(float(prediction), 0.0)
            predictions.append({
                'year': year,
                'privation': privation,
                'country': country,
                'predicted_value': prediction,
            })
            values.append(prediction)
        return pd.DataFrame(predictions)

    def predict_comparative_privations(self, privations: list, country: str, begin_year: int = 2026, years: int = 5):
        results = {privation: self.predict(privation, country, begin_year, years) for privation in privations}
        comparison = pd.DataFrame({'Year': list(range(begin_year, begin_year + years))})
        for privation, frame in results.items():
            comparison[privation] = frame['predicted_value'].values
        return comparison

    def predict_comparative_countries(self, privation: str, countries: list, begin_year: int = 2026, years: int = 5):
        results = {country: self.predict(privation, country, begin_year, years) for country in countries}
        comparison = pd.DataFrame({'Year': list(range(begin_year, begin_year + years))})
        for country, frame in results.items():
            comparison[country] = frame['predicted_value'].values
        return comparison


def predict_contribution_latam(privation: str, country: str, begin_year: int = 2026, years: int = 5):
    return ContributionLatamPredictor().predict(privation, country, begin_year, years)


def predict_comparative_privations(privations: list, country: str, begin_year: int = 2026, years: int = 5):
    return ContributionLatamPredictor().predict_comparative_privations(privations, country, begin_year, years)


def predict_comparative_countries(privation: str, countries: list, begin_year: int = 2026, years: int = 5):
    return ContributionLatamPredictor().predict_comparative_countries(privation, countries, begin_year, years)


if __name__ == '__main__':
    print(ContributionLatamPredictor().predict('Agua', 'Argentina', 2026, 5).to_string(index=False))