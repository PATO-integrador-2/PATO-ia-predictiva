# predict_average_simple.py
"""
Prediccion de porcentaje de privaciones usando un modelo simplificado (Random Forest).
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

class AverageSimplePredictor:
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'models', 'train_lv1_vw3')
        
        # Cargar artefactos
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_domain = joblib.load(os.path.join(self.model_dir, 'label_encoder_domain.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        
        # Cargar métricas
        import json
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r') as f:
            self.metrics = json.load(f)
        
        print("Modelo simplificado cargado")
        print(f"   R²: {self.metrics['r2']:.4f}")
        print(f"   Features: {len(self.feature_cols)}")
    
    #Obtiene datos históricos para un dominio.
    def _get_historical_data(self, domain: str, years_hist: int = 5):
        data_base = Database()
        data_frame = data_base.get_data_view_average_deprivations()
        data_frame = data_frame[data_frame['dominio'] == domain]
        data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)]
        data_frame = data_frame.sort_values('anio')
        
        if len(data_frame) < years_hist:
            raise ValueError(f"Datos insuficientes para {domain}")
        
        return data_frame.tail(years_hist)
    
    #Calcula las 10 features para un año de predicción.
    def _calcular_features(self, df_historical, prediction_year: int):
        # Obtener datos necesarios
        domain = df_historical['dominio'].iloc[0]
        values = df_historical['porcentaje'].values
        years = df_historical['anio'].values
        
        # Últimos valores
        last_value = values[-1]
        penultimate_value = values[-2] if len(values) >= 2 else last_value
        antepenultimate_value = values[-3] if len(values) >= 3 else last_value
        
        # Calcular global_mean (promedio de todos los dominios en ese año)
        data_base = Database()
        df_global = data_base.get_data_view_average_deprivations()
        df_global = df_global[~df_global['dominio'].str.contains('__test__', na=False)]
        
        # 1. global_mean
        global_mean = df_global[df_global['anio'] == prediction_year]['porcentaje'].mean()
        if pd.isna(global_mean):
            # Si no hay datos para ese año, usar el último disponible
            last_year = df_global['anio'].max()
            global_mean = df_global[df_global['anio'] == last_year]['porcentaje'].mean()
        
        # 2. deviation_from_global
        deviation_from_global = last_value - global_mean
        
        # 3. domain_percentile (percentil del dominio en ese año)
        df_year = df_global[df_global['anio'] == prediction_year]
        if len(df_year) > 0:
            domain_percentile = df_year['porcentaje'].rank(pct=True).iloc[0]
        else:
            domain_percentile = 0.5
        
        # 4. lag1 (valor del año anterior)
        lag1 = penultimate_value
        
        # 5. ma2 (media móvil 2 años)
        if len(values) >= 2:
            ma2 = (penultimate_value + antepenultimate_value) / 2
        else:
            ma2 = last_value
        
        # 6. global_ma3 (media móvil global 3 años)
        df_global_3 = df_global[df_global['anio'] >= prediction_year - 3]
        if len(df_global_3) > 0:
            global_ma3 = df_global_3['porcentaje'].mean()
        else:
            global_ma3 = global_mean
        
        # 7. ma3 (media móvil 3 años del dominio)
        if len(values) >= 3:
            ma3 = (values[-1] + values[-2] + values[-3]) / 3
        else:
            ma3 = last_value
        
        # 8. acceleration (aceleración del cambio)
        if len(values) >= 3:
            acceleration = (values[-1] - values[-2]) - (values[-2] - values[-3])
        else:
            acceleration = 0
        
        # 9. pos_relative_max (posición relativa al máximo)
        max_historico = df_historical['porcentaje'].max()
        min_historico = df_historical['porcentaje'].min()
        if max_historico > min_historico:
            pos_relative_max = (last_value - min_historico) / (max_historico - min_historico)
        else:
            pos_relative_max = 0.5
        
        # 10. domain_norm (normalización del dominio)
        mean_historical = df_historical['porcentaje'].mean()
        std_historical = df_historical['porcentaje'].std()
        if std_historical > 0:
            domain_norm = (last_value - mean_historical) / std_historical
        else:
            domain_norm = 0
        
        return {
            'deviation_from_global': deviation_from_global,
            'domain_percentile': domain_percentile,
            'lag1': lag1,
            'ma2': ma2,
            'global_ma3': global_ma3,
            'ma3': ma3,
            'acceleration': acceleration,
            'global_mean': global_mean,
            'pos_relative_max': pos_relative_max,
            'domain_norm': domain_norm
        }
    #Predice el porcentaje de privaciones para un dominio en años futuros.
    def predict(self, domain: str, begin_year: int = 2026, years: int = 5):
        
        # 1. Obtener datos históricos
        historical = self._get_historical_data(domain, years_hist=5)
        
        print(f" Datos históricos para {domain}:")
        for _, row in historical.iterrows():
            print(f"   {int(row['anio'])}: {row['porcentaje']:.2f}%")
        
        # 2. Inicializar valores para predicción recursiva
        df_historical = historical.copy()
        predictions = []
        
        for i in range(years):
            year = begin_year + i
            
            # Calcular features para este año
            features_dict = self._calcular_features(df_historical, year)
            
            # Crear array en el orden correcto
            features = [features_dict[col] for col in self.feature_cols]
            
            # Escalar
            features_scaled = self.scaler.transform([features])
            
            # Predecir
            pred = self.model.predict(features_scaled)[0]
            
            predictions.append({
                'año': year,
                'dominio': domain,
                'porcentaje_privaciones': float(pred)
            })
            
            # Actualizar histórico para la siguiente predicción
            nueva_fila = pd.DataFrame({
                'anio': [year],
                'dominio': [domain],
                'porcentaje': [pred]
            })
            df_historical = pd.concat([df_historical, nueva_fila], ignore_index=True)
            df_historical = df_historical.tail(5)  # Mantener solo los últimos 5
        
        return pd.DataFrame(predictions)
    
    def predict_multiple_domains(self, domains: list, begin_year: int = 2026, years: int = 5):
        """
        Predice para múltiples dominios y genera tabla comparativa.
        """
        results = {}
        
        for domain in domains:
            try:
                df = self.predict(domain, begin_year, years)
                results[domain] = df
                print(f" {domain}: predicciones generadas")
            except Exception as e:
                print(f" {domain}: {e}")
        
        if not results:
            return None
        
        comparison = pd.DataFrame()
        comparison['Año'] = list(range(begin_year, begin_year + years))
        
        for domain, df in results.items():
            comparison[domain] = df['porcentaje_privaciones'].values
        
        return comparison


# FUNCIÓN DE CONVENIENCIA

def predict_porcentaje(domain: str, begin_year: int = 2026, years: int = 5):
    """Función rápida para predecir un dominio."""
    predictor = AverageSimplePredictor()
    return predictor.predict(domain, begin_year, years)


# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    print(" PREDICCIONES - PORCENTAJE DE PRIVACIONES")
    print("-" * 60)
    predictor = AverageSimplePredictor()
    
    # 1. Predicción: Nacional
    print("\n 1. PREDICCIÓN: Nacional")
    print("-" * 40)
    df = predictor.predict('Nacional', begin_year=2026, years=5)
    print("\n RESULTADOS:")
    print(df.to_string(index=False))
    
    # 2. Predicción: Caribe
    print("\n 2. PREDICCIÓN: Caribe")
    print("-" * 40)
    df = predictor.predict('Caribe', begin_year=2026, years=5)
    print("\n RESULTADOS:")
    print(df.to_string(index=False))
    
    # 3. Comparativa de dominios
    print("\n 3. COMPARATIVA DE DOMINIOS")
    print("-" * 40)
    dominios = ['Nacional', 'Caribe', 'Bogotá (Cabecera)']
    comparativa = predictor.predict_multiple_domains(dominios, begin_year=2026, years=5)
    
    if comparativa is not None:
        print("\n TABLA COMPARATIVA:")
        print(comparativa.to_string(index=False))
