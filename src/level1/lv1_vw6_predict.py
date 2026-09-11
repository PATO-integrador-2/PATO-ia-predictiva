"""
Predictor 
Incidencia de Pobreza Multidimensional según sexo del jefe del hogar
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from database import Database

#    Predictor para la incidencia de pobreza por sexo del jefe del hogar.
class IncidenceByHeadSexPredictor:

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'models', 'train_lv1_vw6')

        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_domain = joblib.load(os.path.join(self.model_dir, 'label_encoder_domain.pkl'))
        self.le_sex = joblib.load(os.path.join(self.model_dir, 'label_encoder_sex.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.sexes = joblib.load(os.path.join(self.model_dir, 'sexes.pkl'))
        self.domains = joblib.load(os.path.join(self.model_dir, 'domains.pkl'))

        import json
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r') as f:
            self.metrics = json.load(f)

        print("Modelo de incidencia por sexo del jefe del hogar cargado")
        print(f"   R²: {self.metrics['r2']:.4f}")


#Muestra las opciones disponibles para consultar.
    def show_options(self):

        print("")
        print("OPCIONES DISPONIBLES PARA CONSULTAR")
        print("")

        print("SEXOS DISPONIBLES:")
        for i, sex in enumerate(self.sexes, 1):
            print(f"   {i}. {sex}")

        print("")
        print("DOMINIOS DISPONIBLES:")
        for i, dom in enumerate(self.domains, 1):
            print(f"   {i}. {dom}")

    #Obtiene datos historicos para un dominio y sexo.
    def _get_historical_data(self, domain: str, sex: str):

        database = Database()
        data_frame = database.get_data_view_incidence_by_household_head_sex()

        data_frame = data_frame[data_frame['dominio'] == domain]
        data_frame = data_frame[data_frame['sexo'] == sex]
        data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)]
        data_frame = data_frame.sort_values('anio')

        if len(data_frame) < 4:
            raise ValueError(f"Datos insuficientes para {domain} - {sex}")

        return data_frame.tail(4)

    #Predice la incidencia de pobreza para un sexo en un dominio.

    def predict(self, domain: str, sex: str, begin_year: int = 2026, years: int = 5):

        historical = self._get_historical_data(domain, sex)
        last_values = historical.tail(4)

        historical_years = last_values['anio'].values
        historical_values = last_values['porcentaje'].values

        print("")
        print(f"Datos historicos para {sex} - {domain}:")
        for y, v in zip(historical_years, historical_values):
            print(f"   {y}: {v:.2f}%")

        domain_encoded = self.le_domain.transform([domain])[0]
        sex_encoded = self.le_sex.transform([sex])[0]

        mean_historical = np.mean(historical_values)
        std_historical = np.std(historical_values) if np.std(historical_values) > 0 else 1
        values_norm = (historical_values - mean_historical) / std_historical

        current_year = historical_years[-1]
        current_val_norm = values_norm[-1]
        lag1 = values_norm[-2] if len(values_norm) >= 2 else current_val_norm
        lag2 = values_norm[-3] if len(values_norm) >= 3 else current_val_norm
        lag3 = values_norm[-4] if len(values_norm) >= 4 else current_val_norm

        trend = current_val_norm - lag1

        predictions = []

        for i in range(years):
            year = begin_year + i

            features_dict = {
                'year': year,
                'year_squared': year ** 2,
                'years_since_2010': year - 2010,
                'lag1': current_val_norm,
                'lag2': lag1,
                'lag3': lag2,
                'ma2': (current_val_norm + lag1) / 2 if len(values_norm) >= 2 else current_val_norm,
                'ma3': (current_val_norm + lag1 + lag2) / 3 if len(values_norm) >= 3 else current_val_norm,
                'trend': trend,
                'global_mean': current_val_norm,
                'deviation_from_global': 0,
                'domain_mean': current_val_norm,
                'domain_std': std_historical,
                'domain_norm': current_val_norm,
                'domain_encoded': domain_encoded,
                'sex_encoded': sex_encoded
            }

            features = [features_dict[col] for col in self.feature_cols]
            features_scaled = self.scaler.transform([features])

            pred_norm = self.model.predict(features_scaled)[0]

            pred_real = pred_norm * std_historical + mean_historical
            pred_real = max(pred_real, 0)
            pred_real = min(pred_real, 100)

            predictions.append({
                'year': year,
                'sex': sex,
                'domain': domain,
                'predicted_percentage': float(pred_real)
            })

            lag3 = lag2
            lag2 = lag1
            lag1 = current_val_norm
            current_val_norm = pred_norm
            current_year = year
            trend = current_val_norm - lag1

        return pd.DataFrame(predictions)

    def predict_comparison(self, domain: str, begin_year: int, years: int):
        """Predice y compara ambos sexos para un dominio."""

        results = {}

        for sex in self.sexes:
            try:
                df = self.predict(domain, sex, begin_year, years)
                results[sex] = df
                print(f" {sex}: predicciones generadas")
            except Exception as e:
                print(f" {sex}: {e}")

        if not results:
            return None

        comparison = pd.DataFrame()
        comparison['Year'] = list(range(begin_year, begin_year + years))

        for sex, df in results.items():
            comparison[sex] = df['predicted_percentage'].values

        if len(results) == 2:
            comparison['Difference'] = comparison.iloc[:, 1] - comparison.iloc[:, 2]

        return comparison


def predict_incidence_by_head_sex(domain: str, sex: str, begin_year: int, years: int):
    """Funcion de conveniencia para predecir un sexo en un dominio."""

    predictor = IncidenceByHeadSexPredictor()
    return predictor.predict(domain, sex, begin_year, years)


def predict_comparison_head_sex(domain: str, begin_year: int, years: int):
    """Funcion de conveniencia para comparar ambos sexos."""

    predictor = IncidenceByHeadSexPredictor()
    return predictor.predict_comparison(domain, begin_year, years)


if __name__ == "__main__":

    print("PREDICCIONES - INCIDENCIA DE POBREZA POR SEXO DEL JEFE DEL HOGAR")
    print("")

    predictor = IncidenceByHeadSexPredictor()

    predictor.show_options()

    print("")
    print("1. PREDICCION: Mujer - Nacional")
    print("")
    df = predictor.predict('Nacional', 'Mujer', begin_year=2026, years=5)
    print("")
    print("RESULTADOS:")
    print(df.to_string(index=False))

    print("")
    print("2. COMPARATIVA: Hombre vs Mujer - Nacional")
    print("")
    comparison = predictor.predict_comparison('Nacional', begin_year=2026, years=5)

    if comparison is not None:
        print("")
        print("TABLA COMPARATIVA:")
        print(comparison.round(2).to_string(index=False))