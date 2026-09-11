
# Prediccion de contribucion relativa de privaciones en Latinoamerica.


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from database import Database

# Predictor para contribucion relativa de privaciones en Latinoamerica.

class ContributionLatamPredictor:
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'models', 'train_lv3_vw1')
        
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_country = joblib.load(os.path.join(self.model_dir, 'label_encoder_country.pkl'))
        self.le_privation = joblib.load(os.path.join(self.model_dir, 'label_encoder_privation.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.countries = joblib.load(os.path.join(self.model_dir, 'countries.pkl'))
        self.privations = joblib.load(os.path.join(self.model_dir, 'privations.pkl'))
        
        import json
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r') as file:
            self.metrics = json.load(file)
        
        print("Modelo de contribucion latinoamericana cargado")
        print(f"   R²: {self.metrics['r2']:.4f}")
    
    # Muestra las opciones disponibles para consultar.

    def show_options(self):
        
        print("")
        print("OPCIONES DISPONIBLES PARA CONSULTAR")
        print("")
        
        print("PRIVACIONES DISPONIBLES:")
        for i, priv in enumerate(self.privations, 1):
            print(f"   {i}. {priv}")
        
        print("")
        print("PAISES DISPONIBLES:")
        for i, country in enumerate(self.countries, 1):
            print(f"   {i}. {country}")
    
    # Obtiene datos historicos para una privacion y pais.

    def _get_historical_data(self, privation: str, country: str):
        
        database = Database()
        data_frame = database.get_data_view_dashboard03_deprivation_contribution()
        
        data_frame = data_frame[data_frame['privacion'] == privation]
        data_frame = data_frame[data_frame['pais'] == country]
        data_frame = data_frame.sort_values('anio')
        
        if len(data_frame) < 4:
            raise ValueError(f"Datos insuficientes para {privation} - {country}")
        
        return data_frame.tail(4)
    
    # Predice el valor de una privacion para un pais.
    def predict(self, privation: str, country: str, begin_year: int = 2026, years: int = 5):
        
        historical = self._get_historical_data(privation, country)
        last_values = historical.tail(4)
        
        historical_years = last_values['anio'].values
        historical_values = last_values['valor_porcentaje'].values
        
        print("")
        print(f"Datos historicos para {privation} - {country}:")
        for year, value in zip(historical_years, historical_values):
            print(f"   {year}: {value:.2f}%")
        
        country_encoded = self.le_country.transform([country])[0]
        privation_encoded = self.le_privation.transform([privation])[0]
        
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
                'value_lag1': current_val_norm,
                'value_lag2': lag1,
                'value_lag3': lag2,
                'value_trend': trend,
                'global_mean': current_val_norm,
                'deviation_from_global': 0,
                'country_encoded': country_encoded,
                'privation_encoded': privation_encoded
            }
            
            features = [features_dict[col] for col in self.feature_cols]
            features_scaled = self.scaler.transform([features])
            
            pred_norm = self.model.predict(features_scaled)[0]
            
            pred_real = pred_norm * std_historical + mean_historical
            pred_real = max(pred_real, 0)
            
            predictions.append({
                'year': year,
                'privation': privation,
                'country': country,
                'predicted_value': float(pred_real)
            })
            
            lag3 = lag2
            lag2 = lag1
            lag1 = current_val_norm
            current_val_norm = pred_norm
            current_year = year
            trend = current_val_norm - lag1
        
        return pd.DataFrame(predictions)
    
    # Predice y compara multiples privaciones para un pais.

    def predict_comparative_privations(self, privations: list, country: str, begin_year: int = 2026, years: int = 5):
        
        results = {}
        
        for privation in privations:
            try:
                data_frame = self.predict(privation, country, begin_year, years)
                results[privation] = data_frame
                print(f"   {privation}: predicciones generadas")
            except Exception as error:
                print(f"   {privation}: {error}")
        
        if not results:
            return None
        
        comparison = pd.DataFrame()
        comparison['Year'] = list(range(begin_year, begin_year + years))
        
        for privation, data_frame in results.items():
            comparison[privation] = data_frame['predicted_value'].values
        
        return comparison
    
    # Predice y compara una privacion en multiples paises.

    def predict_comparative_countries(self, privation: str, countries: list, begin_year: int = 2026, years: int = 5):
        
        results = {}
        
        for country in countries:
            try:
                data_frame = self.predict(privation, country, begin_year, years)
                results[country] = data_frame
                print(f"   {country}: predicciones generadas")
            except Exception as error:
                print(f"   {country}: {error}")
        
        if not results:
            return None
        
        comparison = pd.DataFrame()
        comparison['Year'] = list(range(begin_year, begin_year + years))
        
        for country, data_frame in results.items():
            comparison[country] = data_frame['predicted_value'].values
        
        return comparison

    # Funcion de conveniencia para predecir una privacion en un pais.

def predict_contribution_latam(privation: str, country: str, begin_year: int = 2026, years: int = 5):
    
    predictor = ContributionLatamPredictor()
    return predictor.predict(privation, country, begin_year, years)

# Funcion de conveniencia para comparar privaciones en un pais.

def predict_comparative_privations(privations: list, country: str, begin_year: int = 2026, years: int = 5):
    
    predictor = ContributionLatamPredictor()
    return predictor.predict_comparative_privations(privations, country, begin_year, years)

# Funcion de conveniencia para comparar una privacion en paises.

def predict_comparative_countries(privation: str, countries: list, begin_year: int = 2026, years: int = 5):
    
    predictor = ContributionLatamPredictor()
    return predictor.predict_comparative_countries(privation, countries, begin_year, years)


if __name__ == "__main__":
    
    print("PREDICCIONES - CONTRIBUCION RELATIVA DE PRIVACIONES")
    print("")
    
    predictor = ContributionLatamPredictor()
    
    predictor.show_options()
    
    print("")
    print("1. PREDICCION: Acceso a Internet - Argentina")
    print("")
    data_frame = predictor.predict('Acceso a Internet', 'Argentina', begin_year=2026, years=5)
    print("")
    print("RESULTADOS:")
    print(data_frame.to_string(index=False))