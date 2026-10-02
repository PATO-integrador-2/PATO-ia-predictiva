"""
Permite predecir IPM por año y dominio específico.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import json
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
        try:
            self.arima_models = joblib.load(os.path.join(models_dir, 'arima_models.pkl'))
        except FileNotFoundError:
            self.arima_models = {}
        self.model_dir = models_dir
        
        # Cargar métricas (opcional)
        try:
            with open(os.path.join(models_dir, 'metrics.json'), 'r') as f:
                self.metrics = json.load(f)
            random_forest_metrics = self.metrics.get('candidates', {}).get(
                'random_forest', self.metrics.get('random_forest', self.metrics)
            )
            r2 = random_forest_metrics.get('r2')
            print(f"Modelo cargado (R²: {r2:.4f})" if r2 is not None else "Modelo cargado")
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
        
        data_frame['anio'] = pd.to_numeric(data_frame['anio'], errors='raise').astype(int)
        data_frame = data_frame.drop_duplicates(subset=['dominio', 'anio'])
        data_frame = data_frame.sort_values('anio')
        if len(data_frame) < 3:
            raise ValueError(f"Datos insuficientes para {dominio} (solo {len(data_frame)} años)")

        recent_data = data_frame.tail(3)
        recent_years = recent_data['anio'].to_numpy()
        if not np.all(np.diff(recent_years) == 1):
            raise ValueError(
                f"La ventana reciente de {dominio} contiene años faltantes: "
                f"{recent_years.tolist()}"
            )

        all_years = data_frame['anio'].to_numpy()
        if not np.all(np.diff(all_years) == 1):
            print(
                f"Advertencia: {dominio} tiene huecos históricos; "
                f"se usará la ventana consecutiva {recent_years.tolist()}"
            )

        return recent_data


    def predict(self, domain: str, begin_year: int, years: int = 5):
        
        # 1. Obtener datos históricos del dominio
        historical = self._get_historical_data(domain)
        
        # 2. Extraer últimos valores
        last_values = historical.tail(3).copy()
        historical_years = last_values['anio'].values
        historical_ipms = last_values['ipm'].values
        
        # 3. Codificar dominio
        domain_encoded = self.le.transform([domain])[0]
        
        # 4. Inicializar la ventana histórica para el pronóstico temporal
        current_year = historical_years[-1]
        current_ipm = historical_ipms[-1]
        lag1 = historical_ipms[-2] if len(historical_ipms) >= 2 else current_ipm
        lag2 = historical_ipms[-3] if len(historical_ipms) >= 3 else current_ipm
        changes = list(np.diff(historical_ipms))
        trend = float(np.mean(changes[-3:])) if changes else 0.0
        selected_model = self.metrics.get('selected_model', 'RandomForestRegressor')
        
        # 5. Hacer predicciones recursivas
        predicciones = []
        
        for i in range(years):
            year = begin_year + i
            
            if selected_model == 'naive':
                pred = current_ipm
            elif selected_model == 'arima':
                arima_model = self.arima_models.get(domain)
                pred = (
                    arima_model.forecast(steps=i + 1)[-1]
                    if arima_model is not None else current_ipm
                )
            else:
                features = {
                    'year': year,
                    'years_since_2010': year - 2010,
                    'ipm_lag1': current_ipm,
                    'ipm_lag2': lag1,
                    'domain_trend': trend,
                    'domain_encoded': domain_encoded,
                }
                feature_values = [features[col] for col in self.feature_cols]
                features_scaled = self.scaler.transform([feature_values])
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
            changes.append(current_ipm - lag1)
            trend = float(np.mean(changes[-3:]))
        
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
