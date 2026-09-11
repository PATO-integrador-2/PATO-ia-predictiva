
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from database import Database


class PrivationsPredictor:
    """
    Predictor para el modelo de privaciones.
    """
    
    def __init__(self, usar_global: bool = True):
        self.usar_global = usar_global
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.global_dir = os.path.join(self.base_dir, 'models', 'train_lv1_vw2')
        
        # Cargar archivos
        self.model = joblib.load(os.path.join(self.global_dir, 'model.pkl'))
        self.scaler_global = joblib.load(os.path.join(self.global_dir, 'scaler_global.pkl'))
        self.le_domain = joblib.load(os.path.join(self.global_dir, 'label_encoder_domain.pkl'))
        self.le_variable = joblib.load(os.path.join(self.global_dir, 'label_encoder_variable.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.global_dir, 'feature_cols.pkl'))
        
        print(" Modelo global cargado correctamente")
        print(f"   Features: {len(self.feature_cols)} columnas")
    
    # Obtiene los últimos datos históricos para una variable y dominio.
    def _get_historical_data(self, variable: str, domain: str):
        
        database = Database()
        data_frame = database.get_data_view_deprivations_variable()
        
        data_frame = data_frame[data_frame['variable'] == variable]
        data_frame = data_frame[data_frame['dominio'] == domain]
        data_frame = data_frame.sort_values('anio')
        
        if len(data_frame) < 4:
            raise ValueError(f"Datos insuficientes para {variable} - {domain}")
        
        return data_frame.tail(4)
    
    # Convierte una predicción escalada a su valor real usando los históricos.
    
    def rescale_prediction(self, scaled_value: float, historical_ipms: np.ndarray) -> float:
        # Usar la media y desviación de los valores históricos reales
        mean_historical = np.mean(historical_ipms)
        std_historical = np.std(historical_ipms) if np.std(historical_ipms) > 0 else 1
        
        # Desescalar: valor_real = valor_escalado * std + mean
        real_value = scaled_value * std_historical + mean_historical
        
        # Asegurar que no sea negativo
        return float(max(real_value, 0))
    
    # Normaliza los valores de IPM manualmente (sin usar scaler_global).
    
    def _normalize_ipm(self, valores: np.ndarray) -> np.ndarray:
        mean_val = np.mean(valores)
        std_val = np.std(valores) if np.std(valores) > 0 else 1
        return (valores - mean_val) / std_val
    
    # Predice el IPM para una variable y dominio en años futuros.
    def predict(self, variable: str, domain: str, begin_year: int = 2026, years: int = 5):

        # 1. Obtener datos históricos
        historical = self._get_historical_data(variable, domain)
        last_4 = historical.tail(4)
        
        historical_years = last_4['anio'].values
        historical_ipms = last_4['ipm'].values
        
        print(f"-  Datos históricos para {variable} - {domain}:")
        for a, i in zip(historical_years, historical_ipms):
            print(f"   {a}: {i:.2f}%")
        
        # 2. Codificar dominio y variable
        domain_encoded = self.le_domain.transform([domain])[0]
        variable_encoded = self.le_variable.transform([variable])[0]
        
        # 3. NORMALIZAR IPM MANUALMENTE
        normalized_ipms = self._normalize_ipm(historical_ipms)
        
        # 4. Inicializar variables para predicción recursiva
        current_year = historical_years[-1]
        current_ipm_norm = normalized_ipms[-1]
        lag1 = normalized_ipms[-2] if len(normalized_ipms) >= 2 else current_ipm_norm
        lag2 = normalized_ipms[-3] if len(normalized_ipms) >= 3 else current_ipm_norm
        lag3 = normalized_ipms[-4] if len(normalized_ipms) >= 4 else current_ipm_norm
        
        domain_trend = current_ipm_norm - lag1
        variable_trend = current_ipm_norm - lag1
        
        predictions = []
        
        for i in range(years):
            year = begin_year + i
            
            # Construir features en el orden correcto
            features_dict = {
                'year': year,
                'year_squared': year ** 2,
                'years_since_2010': year - 2010,
                'ipm_lag1': current_ipm_norm,
                'ipm_lag2': lag1,
                'ipm_lag3': lag2,
                'domain_trend': domain_trend,
                'variable_trend': variable_trend,
                'domain_encoded': domain_encoded,
                'variable_encoded': variable_encoded
            }
            
            # Crear array en el orden de feature_cols
            features = [features_dict[col] for col in self.feature_cols]
            
            # ESCALAR FEATURES (esto sí usa scaler_global)
            features_scaled = self.scaler_global.transform([features])
            
            # Predecir
            normalized_prediction = self.model.predict(features_scaled)[0]
            
            # Desescalar usando los históricos
            real_pred = self.rescale_prediction(normalized_prediction, historical_ipms)
            
            predictions.append({
                'year': year,
                'variable': variable,
                'domain': domain,
                'ipm_foretold': real_pred
            })
            
            # Actualizar recursividad
            lag3 = lag2
            lag2 = lag1
            lag1 = current_ipm_norm
            current_ipm_norm = normalized_prediction
            current_year = year
            domain_trend = current_ipm_norm - lag1
            variable_trend = current_ipm_norm - lag1
        
        return pd.DataFrame(predictions)


# FUNCIÓN DE CONVENIENCIA

def predict_privacion(variable: str, domain: str, begin_year: int = 2026, years: int = 5):
    predictor = PrivationsPredictor(usar_global=True)
    return predictor.predict(variable, domain, begin_year, years)


# Ejecución principal para pruebas

if __name__ == "__main__":

    print("PREDICCIONES - MODELO DE PRIVACIONES")
    
    predictor = PrivationsPredictor(usar_global=True)
    
    # 1. Predicción: Analfabetismo - Nacional
    print("\n  1. PREDICCIÓN: Analfabetismo - Nacional")
    print("-" * 40)
    df = predictor.predict('Privación por Analfabetismo', 'Nacional', 2026, 5)
    print("\n  RESULTADOS:")
    print(df.to_string(index=False))
    
    # 2. Predicción: Inasistencia escolar - Nacional
    print("\n  2. PREDICCIÓN: Inasistencia Escolar - Nacional")
    print("-" * 40)
    df = predictor.predict('Privación por Inasistencia Escolar', 'Nacional', 2026, 5)
    print("\n  RESULTADOS:")
    print(df.to_string(index=False))