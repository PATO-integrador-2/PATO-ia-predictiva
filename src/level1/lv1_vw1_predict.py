"""
Permite predecir IPM por año y dominio específico.
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


class DomainPredictor:
    
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root =os.path.dirname(os.path.dirname(current_dir))
        models_dir = os.path.join(project_root, 'models', 'train_lv1_vw1')
        self.model = joblib.load(os.path.join(models_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        self.le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
        
        # Cargar métricas (opcional)
        try:
            import json
            with open(os.path.join(models_dir, 'metrics.json'), 'r') as f:
                self.metrics = json.load(f)
            print(f"Modelo cargado (R²: {self.metrics['r2']:.4f})")
        except:
            print("Modelo cargado")

    
    def _get_historical_data(self, dominio: str):
        # Obtener los últimos 3 años históricos del dominio.

        database = Database()
        data_frame = database.get_data_view_ipm_domain()
        
        # Filtrar por dominio y limpiar
        data_frame = data_frame[data_frame['dominio'] == dominio]
        data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)]
        data_frame = data_frame.sort_values('anio')
        
        if len(data_frame) < 3:
            raise ValueError(f"Datos insuficientes para {dominio} (solo {len(data_frame)} años)")
        
        return data_frame.tail(3)


    def predict(self, domain: str, begin_year: int, years: int = 5):
        
        # 1. Obtener datos históricos del dominio
        historical = self._get_historical_data(domain)
        
        # 2. Extraer últimos valores
        last_values = historical.tail(3).copy()
        historical_years = last_values['anio'].values
        historical_ipms = last_values['ipm'].values
        
        # 3. Codificar dominio
        domain_encoded = self.le.transform([domain])[0]
        
        # 4. Inicializar variables para predicción recursiva
        # Usamos los últimos 3 años históricos
        current_year = historical_years[-1]
        current_ipm = historical_ipms[-1]
        lag1 = historical_ipms[-2] if len(historical_ipms) >= 2 else current_ipm
        lag2 = historical_ipms[-3] if len(historical_ipms) >= 3 else current_ipm
        trend = current_ipm - lag1 if len(historical_ipms) >= 2 else 0
        
        # 5. Hacer predicciones recursivas
        predicciones = []
        
        for i in range(years):
            year = begin_year + i
            
            # Preparar features (mismo orden que en entrenamiento)
            features = [
                year,
                year ** 2,
                year - 2010,
                current_ipm,   # lag1 (IPM más reciente)
                lag1,          # lag2 (IPM anterior)
                trend,         # tendencia del dominio
                domain_encoded
            ]
            
            # Escalar
            features_scaled = self.scaler.transform([features])
            
            # Predecir
            pred = self.model.predict(features_scaled)[0]
            
            predicciones.append({
                'año': year,
                'dominio': domain,
                'ipm_predicho': float(pred)
            })
            
            # Actualizar para la siguiente iteración (recursividad)
            lag2 = lag1
            lag1 = current_ipm
            current_ipm = pred
            current_year = year
            trend = current_ipm - lag1 if len(predicciones) >= 2 else trend
        
        return pd.DataFrame(predicciones)

    """
    Predecir múltiples dominios al tiempo
    """

    def predict_multiple(self, domains: list, begin_year: int = 2026, years: int = 5):
        
        results = {}
        
        for domain in domains:
            try:
                data_frame = self.predict(domain, begin_year, years)
                results[domain] = data_frame
                print(f"{domain}: predicciones generadas")
            except Exception as e:
                print(f"Error en {domain}: {e}")
        
        if not results:
            return None
        
        comparison = pd.DataFrame()
        for domain, data_frame in results.items():
            comparison[domain] = data_frame['ipm_predicho'].values
        
        return comparison


# Funciones
# Función para predecir un dominio.
def predict_ipm(dominio: str, begin_year: int = 2026, years: int = 5):
    predictor = DomainPredictor()
    return predictor.predict(dominio, begin_year, years)

#Función para comparar dominios.
def predict_comparative(dominios: list, begin_year: int = 2026, years: int = 5):
    
    predictor = DomainPredictor()
    return predictor.predict_multiple(dominios, begin_year, years)


# Ejecución

if __name__ == "__main__":

    print("PREDICCIONES - VISTA 1 (PRUEBA)")
    
    # 1. Crear predictor
    predictor = DomainPredictor()
    
    # 2. Predicción para Nacional
    print("\n- PREDICCIÓN: Nacional")
    print("-" * 40)
    df_nac = predictor.predict('Nacional', begin_year=2026, years=5)
    print(df_nac.to_string(index=False))
    
    # 3. Predicción para Caribe
    print("\n- PREDICCIÓN: Caribe")
    print("-" * 40)
    df_caribe = predictor.predict('Caribe', begin_year=2026, years=5)
    print(df_caribe.to_string(index=False))
    
    # 4. Predicción para Bogotá
    print("\n- PREDICCIÓN: Bogotá (cabecera)")
    print("-" * 40)
    df_bogota = predictor.predict('Bogotá (cabecera)', begin_year=2026, years=5)
    print(df_bogota.to_string(index=False))
    
    # 5. Comparativa múltiple
    print("\n- COMPARATIVA DE DOMINIOS")
    print("-" * 40)
    dominios = ['Nacional', 'Caribe', 'Bogotá (cabecera)', 'Pacífica']
    comparativa = predictor.predict_multiple(dominios, begin_year=2026, years=5)
    
    if comparativa is not None:
        print("\n- TABLA COMPARATIVA:")
        print(comparativa.to_string(index=False))
