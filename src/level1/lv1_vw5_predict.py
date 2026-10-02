import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Database


class IncidenceBySexPredictor:
    """Pronostica incidencia de pobreza por dominio y sexo."""

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(base_dir, 'models', 'train_lv1_vw5')
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        try:
            self.candidate_models = joblib.load(os.path.join(self.model_dir, 'candidate_models.pkl'))
        except FileNotFoundError:
            self.candidate_models = {}
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_domain = joblib.load(os.path.join(self.model_dir, 'label_encoder_domain.pkl'))
        self.le_sex = joblib.load(os.path.join(self.model_dir, 'label_encoder_sex.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.sexes = joblib.load(os.path.join(self.model_dir, 'sexes.pkl'))
        self.domains = joblib.load(os.path.join(self.model_dir, 'domains.pkl'))
        try:
            self.arima_models = joblib.load(os.path.join(self.model_dir, 'arima_models.pkl'))
        except FileNotFoundError:
            self.arima_models = {}
        with open(os.path.join(self.model_dir, 'metrics.json'), encoding='utf-8') as file:
            self.metrics = json.load(file)

    def _get_historical_data(self, domain: str, sex: str):
        frame = Database().get_data_view_incidence_by_person_sex()
        frame = frame[(frame['dominio'] == domain) & (frame['sexo'] == sex)].copy()
        frame['anio'] = pd.to_numeric(frame['anio'], errors='raise').astype(int)
        frame['porcentaje'] = pd.to_numeric(frame['porcentaje'], errors='raise')
        frame = frame.drop_duplicates(subset=['dominio', 'sexo', 'anio']).sort_values('anio')
        if len(frame) < 4:
            raise ValueError(f'Datos insuficientes para {domain} - {sex}')
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

    def _predict_tree(self, model, values, year, domain_encoded, sex_encoded):
        features = {
            'year': year,
            'year_squared': year ** 2,
            'years_since_2010': year - 2010,
            'lag1': values[-1],
            'lag2': values[-2],
            'lag3': values[-3],
            'ma2': float(np.mean(values[-2:])),
            'ma3': float(np.mean(values[-3:])),
            'trend': self._trend(values),
            'domain_encoded': domain_encoded,
            'sex_encoded': sex_encoded,
        }
        feature_frame = pd.DataFrame(
            [[features[column] for column in self.feature_cols]],
            columns=self.feature_cols
        )
        return float(model.predict(self.scaler.transform(feature_frame))[0])

    def predict(self, domain: str, sex: str, begin_year: int = 2026, years: int = 5):
        historical = self._get_historical_data(domain, sex)
        values = historical['porcentaje'].astype(float).tolist()
        domain_encoded = self.le_domain.transform([domain])[0]
        sex_encoded = self.le_sex.transform([sex])[0]
        series_key = f'{domain}::{sex}'
        selected_model = self.metrics.get('selected_model_by_series', {}).get(
            series_key, self.metrics.get('selected_model', 'random_forest')
        )
        arima_model = self.arima_models.get((domain, sex))
        predictions = []

        for offset in range(years):
            year = begin_year + offset
            if selected_model == 'naive':
                prediction = values[-1]
            elif selected_model == 'arima' and arima_model is not None:
                prediction = float(arima_model.forecast(steps=offset + 1)[-1])
            else:
                model = self.candidate_models.get(selected_model, self.model)
                prediction = self._predict_tree(model, values[-3:], year, domain_encoded, sex_encoded)
            prediction = max(0.0, min(float(prediction), 100.0))
            predictions.append({
                'year': year,
                'sex': sex,
                'domain': domain,
                'predicted_percentage': prediction,
            })
            values.append(prediction)
        return pd.DataFrame(predictions)

    def predict_comparison(self, domain: str, begin_year: int = 2026, years: int = 5):
        results = {sex: self.predict(domain, sex, begin_year, years) for sex in self.sexes}
        comparison = pd.DataFrame({'Year': list(range(begin_year, begin_year + years))})
        for sex, frame in results.items():
            comparison[sex] = frame['predicted_percentage'].values
        if len(results) == 2:
            comparison['Difference'] = comparison[self.sexes[0]] - comparison[self.sexes[1]]
        return comparison


def predict_incidence_by_sex(domain: str, sex: str, begin_year: int = 2026, years: int = 5):
    return IncidenceBySexPredictor().predict(domain, sex, begin_year, years)


def predict_comparison(domain: str, begin_year: int = 2026, years: int = 5):
    return IncidenceBySexPredictor().predict_comparison(domain, begin_year, years)


if __name__ == '__main__':
    print(IncidenceBySexPredictor().predict('Nacional', 'Mujer', 2026, 5).to_string(index=False))