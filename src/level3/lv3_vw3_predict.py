"""
Prediccion de incidencia de pobreza multidimensional por grupo de edad y area.
Version con promedios para predicciones mas estables.
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


class PovertyByAgePredictorV2:
    """
    Predictor para incidencia de pobreza por grupo de edad y area.
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'models', 'train_lv3_vw3')
        
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_area = joblib.load(os.path.join(self.model_dir, 'label_encoder_area.pkl'))
        self.le_country = joblib.load(os.path.join(self.model_dir, 'label_encoder_country.pkl'))
        self.le_age_range = joblib.load(os.path.join(self.model_dir, 'label_encoder_age_range.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.areas = joblib.load(os.path.join(self.model_dir, 'areas.pkl'))
        self.countries = joblib.load(os.path.join(self.model_dir, 'countries.pkl'))
        self.age_ranges = joblib.load(os.path.join(self.model_dir, 'age_ranges.pkl'))
        
        import json
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r') as file:
            self.metrics = json.load(file)
        
        print("Modelo de pobreza por edad y area V2 cargado")
        print(f"   R²: {self.metrics['r2']:.4f}")
    
    def show_options(self):
        """Muestra las opciones disponibles para consultar."""
        
        print("")
        print("OPCIONES DISPONIBLES PARA CONSULTAR")
        print("")
        
        print("RANGOS DE EDAD DISPONIBLES:")
        for i, age_range in enumerate(self.age_ranges, 1):
            print(f"   {i}. {age_range}")
        
        print("")
        print("AREAS DISPONIBLES:")
        for i, area in enumerate(self.areas, 1):
            print(f"   {i}. {area}")
        
        print("")
        print("PAISES DISPONIBLES:")
        for i, country in enumerate(self.countries, 1):
            print(f"   {i}. {country}")
    
    def _get_historical_data(self, country: str, area: str, age_range: str):
        """Obtiene datos historicos para un pais, area y rango de edad."""
        
        database = Database()
        data_frame = database.get_data_view_dashboard03_poverty_by_age()
        
        data_frame = data_frame[data_frame['pais'] == country]
        data_frame = data_frame[data_frame['area_geografica'] == area]
        data_frame = data_frame[data_frame['grupo_erario'] == age_range]
        data_frame = data_frame.sort_values('anio')
        
        if len(data_frame) < 5:
            print("")
            print(f"ADVERTENCIA: Datos insuficientes para {country} - {area} - {age_range}")
            raise ValueError(f"Datos insuficientes para {country} - {area} - {age_range}")
        
        return data_frame.tail(5)
    
    def predict(self, country: str, area: str, age_range: str, begin_year: int = 2026, years: int = 5):
        """
        Predice el valor de pobreza para un pais, area y rango de edad.
        """
        
        historical = self._get_historical_data(country, area, age_range)
        last_values = historical.tail(5)
        
        historical_years = last_values['anio'].values
        historical_values = last_values['valor_porcentaje'].values
        
        print("")
        print(f"Datos historicos para {age_range} - {country} ({area}):")
        for year, value in zip(historical_years, historical_values):
            print(f"   {year}: {value:.2f}%")
        
        area_encoded = self.le_area.transform([area])[0]
        country_encoded = self.le_country.transform([country])[0]
        age_range_encoded = self.le_age_range.transform([age_range])[0]
        
        mean_historical = np.mean(historical_values)
        std_historical = np.std(historical_values) if np.std(historical_values) > 0 else 1
        values_norm = (historical_values - mean_historical) / std_historical
        
        current_year = historical_years[-1]
        current_val_norm = values_norm[-1]
        lag1 = values_norm[-2] if len(values_norm) >= 2 else current_val_norm
        lag2 = values_norm[-3] if len(values_norm) >= 3 else current_val_norm
        
        # Promedios de años anteriores
        avg_prev_2 = np.mean(values_norm[-3:-1]) if len(values_norm) >= 3 else current_val_norm
        avg_prev_3 = np.mean(values_norm[-4:-1]) if len(values_norm) >= 4 else current_val_norm
        
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
                'avg_previous_2': avg_prev_2,
                'avg_previous_3': avg_prev_3,
                'value_trend': trend,
                'global_mean': current_val_norm,
                'deviation_from_global': 0,
                'area_encoded': area_encoded,
                'country_encoded': country_encoded,
                'age_range_encoded': age_range_encoded
            }
            
            features = [features_dict[col] for col in self.feature_cols]
            features_scaled = self.scaler.transform([features])
            
            pred_norm = self.model.predict(features_scaled)[0]
            
            pred_real = pred_norm * std_historical + mean_historical
            
            # Limitar entre 0 y 100
            pred_real = max(0, min(pred_real, 100))
            
            # Limitar cambio anual al 10%
            if i > 0:
                previous = predictions[-1]['predicted_value']
                max_change = previous * 0.10
                if pred_real > previous + max_change:
                    pred_real = previous + max_change
                if pred_real < previous - max_change:
                    pred_real = previous - max_change
            
            predictions.append({
                'year': year,
                'country': country,
                'area': area,
                'age_range': age_range,
                'predicted_value': float(pred_real)
            })
            
            # Actualizar para recursividad
            lag2 = lag1
            lag1 = current_val_norm
            current_val_norm = pred_norm
            current_year = year
            trend = current_val_norm - lag1
            avg_prev_2 = np.mean([lag1, lag2]) if len(values_norm) >= 2 else current_val_norm
            avg_prev_3 = np.mean([current_val_norm, lag1, lag2]) if len(values_norm) >= 3 else current_val_norm
        
        return pd.DataFrame(predictions)
    
    def predict_comparative_age_ranges(self, country: str, area: str, age_ranges: list, begin_year: int = 2026, years: int = 5):
        """Predice y compara multiples rangos de edad para un pais y area."""
        
        results = {}
        
        for age_range in age_ranges:
            try:
                data_frame = self.predict(country, area, age_range, begin_year, years)
                results[age_range] = data_frame
                print(f"   {age_range}: predicciones generadas")
            except Exception as error:
                print(f"   {age_range}: {error}")
        
        if not results:
            return None
        
        comparison = pd.DataFrame()
        comparison['Year'] = list(range(begin_year, begin_year + years))
        
        for age_range, data_frame in results.items():
            comparison[age_range] = data_frame['predicted_value'].values
        
        return comparison


def predict_poverty_by_age_v2(country: str, area: str, age_range: str, begin_year: int = 2026, years: int = 5):
    """Funcion de conveniencia para predecir un rango de edad en un pais y area."""
    
    predictor = PovertyByAgePredictorV2()
    return predictor.predict(country, area, age_range, begin_year, years)


def predict_comparative_age_ranges_v2(country: str, area: str, age_ranges: list, begin_year: int = 2026, years: int = 5):
    """Funcion de conveniencia para comparar rangos de edad en un pais y area."""
    
    predictor = PovertyByAgePredictorV2()
    return predictor.predict_comparative_age_ranges(country, area, age_ranges, begin_year, years)


if __name__ == "__main__":
    
    print("PREDICCIONES - POBREZA MULTIDIMENSIONAL POR EDAD Y AREA V2")
    print("")
    
    predictor = PovertyByAgePredictorV2()
    
    predictor.show_options()
    
    print("")
    print("1. PREDICCION: 0 a 17 años - Latin America (weighted average) - National")
    print("")
    data_frame = predictor.predict('Latin America (weighted average)', 'National', '0 a 17 años', begin_year=2026, years=5)
    print("")
    print("RESULTADOS:")
    print(data_frame.to_string(index=False))
    
    print("")
    print("2. COMPARATIVA: Todos los rangos - Latin America (weighted average) - National")
    print("")
    age_ranges = ['0 a 17 años', '18 a 59 años', '60 años y más', 'Total']
    comparativa = predictor.predict_comparative_age_ranges('Latin America (weighted average)', 'National', age_ranges, begin_year=2026, years=5)
    
    if comparativa is not None:
        print("")
        print("TABLA COMPARATIVA:")
        print(comparativa.round(2).to_string(index=False)) 