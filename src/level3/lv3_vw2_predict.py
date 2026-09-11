"""
Prediccion de poblacion en situacion de pobreza multidimensional (cifras nacionales).
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


class NationalPovertyPredictor:
    """
    Predictor para poblacion en pobreza multidimensional en Latinoamerica.
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'models', 'train_lv3_vw2')
        
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_area = joblib.load(os.path.join(self.model_dir, 'label_encoder_area.pkl'))
        self.le_country = joblib.load(os.path.join(self.model_dir, 'label_encoder_country.pkl'))
        self.le_measure = joblib.load(os.path.join(self.model_dir, 'label_encoder_measure.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.areas = joblib.load(os.path.join(self.model_dir, 'areas.pkl'))
        self.countries = joblib.load(os.path.join(self.model_dir, 'countries.pkl'))
        self.measures = joblib.load(os.path.join(self.model_dir, 'measures.pkl'))
        
        import json
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r') as file:
            self.metrics = json.load(file)
        
        print("Modelo de pobreza nacional latinoamericano cargado")
        print(f"   R²: {self.metrics['r2']:.4f}")
    
    def show_options(self):
        """Muestra las opciones disponibles para consultar."""
        
        print("")
        print("OPCIONES DISPONIBLES PARA CONSULTAR")
        print("")
        
        print("MEDIDAS DISPONIBLES:")
        for i, measure in enumerate(self.measures, 1):
            print(f"   {i}. {measure}")
        
        print("")
        print("AREAS DISPONIBLES:")
        for i, area in enumerate(self.areas, 1):
            print(f"   {i}. {area}")
        
        print("")
        print("PAISES DISPONIBLES:")
        for i, country in enumerate(self.countries, 1):
            print(f"   {i}. {country}")
    
    def _get_historical_data(self, country: str, area: str, measure: str):
        """Obtiene datos historicos para un pais, area y medida."""
        
        database = Database()
        data_frame = database.get_data_view_dashboard03_national_poverty()
        
        data_frame = data_frame[data_frame['pais'] == country]
        data_frame = data_frame[data_frame['area_geografica'] == area]
        data_frame = data_frame[data_frame['tipo_medida_pm'] == measure]
        data_frame = data_frame.sort_values('anio')
        
        if len(data_frame) < 4:
            raise ValueError(f"Datos insuficientes para {country} - {area} - {measure}")
        
        return data_frame.tail(4)
    
    def predict(self, country: str, area: str, measure: str, begin_year: int = 2026, years: int = 5):
        """
        Predice el valor de una medida de pobreza para un pais y area.
        """
        
        historical = self._get_historical_data(country, area, measure)
        last_values = historical.tail(4)
        
        historical_years = last_values['anio'].values
        historical_values = last_values['valor_porcentaje'].values
        
        print("")
        print(f"Datos historicos para {measure} - {country} ({area}):")
        for year, value in zip(historical_years, historical_values):
            print(f"   {year}: {value:.2f}%")
        
        area_encoded = self.le_area.transform([area])[0]
        country_encoded = self.le_country.transform([country])[0]
        measure_encoded = self.le_measure.transform([measure])[0]
        
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
                'area_encoded': area_encoded,
                'country_encoded': country_encoded,
                'measure_encoded': measure_encoded
            }
            
            features = [features_dict[col] for col in self.feature_cols]
            features_scaled = self.scaler.transform([features])
            
            pred_norm = self.model.predict(features_scaled)[0]
            
            pred_real = pred_norm * std_historical + mean_historical
            pred_real = max(pred_real, 0)
            
            predictions.append({
                'year': year,
                'country': country,
                'area': area,
                'measure': measure,
                'predicted_value': float(pred_real)
            })
            
            lag3 = lag2
            lag2 = lag1
            lag1 = current_val_norm
            current_val_norm = pred_norm
            current_year = year
            trend = current_val_norm - lag1
        
        return pd.DataFrame(predictions)
    
    def predict_comparative_measures(self, country: str, area: str, measures: list, begin_year: int = 2026, years: int = 5):
        """Predice y compara multiples medidas para un pais y area."""
        
        results = {}
        
        for measure in measures:
            try:
                data_frame = self.predict(country, area, measure, begin_year, years)
                results[measure] = data_frame
                print(f"   {measure}: predicciones generadas")
            except Exception as error:
                print(f"   {measure}: {error}")
        
        if not results:
            return None
        
        comparison = pd.DataFrame()
        comparison['Year'] = list(range(begin_year, begin_year + years))
        
        for measure, data_frame in results.items():
            comparison[measure] = data_frame['predicted_value'].values
        
        return comparison
    
    def predict_comparative_countries(self, countries: list, area: str, measure: str, begin_year: int = 2026, years: int = 5):
        """Predice y compara una medida en multiples paises y una misma area."""
        
        results = {}
        
        for country in countries:
            try:
                data_frame = self.predict(country, area, measure, begin_year, years)
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


def predict_national_poverty(country: str, area: str, measure: str, begin_year: int = 2026, years: int = 5):
    """Funcion de conveniencia para predecir una medida en un pais y area."""
    
    predictor = NationalPovertyPredictor()
    return predictor.predict(country, area, measure, begin_year, years)


def predict_comparative_measures(country: str, area: str, measures: list, begin_year: int = 2026, years: int = 5):
    """Funcion de conveniencia para comparar medidas en un pais y area."""
    
    predictor = NationalPovertyPredictor()
    return predictor.predict_comparative_measures(country, area, measures, begin_year, years)


def predict_comparative_countries(countries: list, area: str, measure: str, begin_year: int = 2026, years: int = 5):
    """Funcion de conveniencia para comparar una medida en paises."""
    
    predictor = NationalPovertyPredictor()
    return predictor.predict_comparative_countries(countries, area, measure, begin_year, years)


if __name__ == "__main__":
    
    print("PREDICCIONES - POBLACION EN POBREZA MULTIDIMENSIONAL")
    print("")
    
    predictor = NationalPovertyPredictor()
    
    # Diagnosticar datos disponibles
    database = Database()
    data_frame = database.get_data_view_dashboard03_national_poverty()

    # Filtrar por país
    df_country = data_frame[data_frame['pais'] == 'Costa Rica']
    print("Datos disponibles para Costa Rica:")
    print(df_country[['anio', 'area_geografica', 'tipo_medida_pm', 'valor_porcentaje']].to_string())

    print("")
    predictor.show_options()
    
    print("")
    print("1. PREDICCION: Incidencia (H) - Costa Rica - National")
    print("")
    data_frame = predictor.predict('Costa Rica', 'National', 'Incidencia (H)', begin_year=2026, years=5)
    print("")
    print("RESULTADOS:")
    print(data_frame.to_string(index=False))
    
    