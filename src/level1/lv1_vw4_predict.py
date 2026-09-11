"""
Predictor de IPM
Contribuciones a la incidencia Ajustada del IPM
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


class DimensionContributionPredictor:
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'models', 'train_lv1_vw4')
        
        # Cargar artefactos
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_domain = joblib.load(os.path.join(self.model_dir, 'label_encoder_domain.pkl'))
        self.le_dimension = joblib.load(os.path.join(self.model_dir, 'label_encoder_dimension.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        self.dimensions = joblib.load(os.path.join(self.model_dir, 'dimensions.pkl'))
        self.domains = joblib.load(os.path.join(self.model_dir, 'domains.pkl'))
        
        # Cargar métricas
        import json
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r') as f:
            self.metrics = json.load(f)
        
        print("Modelo de contribuciones por dimensión cargado")
        print(f"   R²: {self.metrics['r2']:.4f}")
    
    def show_options(self):
        print(" OPCIONES DISPONIBLES PARA CONSULTAR")
        
        print("\nDIMENSIONES DISPONIBLES:")
        for i, dim in enumerate(self.dimensions, 1):
            print(f"   {i}. {dim}")
        
        print("\nDOMINIOS DISPONIBLES:")
        for i, dom in enumerate(self.domains, 1):
            print(f"   {i}. {dom}")
        
    
    def _get_historical_data(self, dimension: str, domain: str):
        """Obtiene datos históricos para una dimensión y dominio."""
        database = Database()
        data_frame = database.get_data_view_dimension_contribution()
        data_frame = data_frame[data_frame['dimension'] == dimension]
        data_frame = data_frame[data_frame['dominio'] == domain]
        data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)]
        data_frame = data_frame.sort_values('anio')
        
        if len(data_frame) < 4:
            raise ValueError(f"Datos insuficientes para {dimension} - {domain}")
        
        return data_frame.tail(4)
    
    def predict(self, dimension: str, domain: str, begin_year: int , years: int ):
        # 1. Obtener datos históricos
        historical = self._get_historical_data(dimension, domain)
        last_values = historical.tail(4)
        
        historical_years = last_values['anio'].values
        historical_values = last_values['porcentaje'].values
        
        print(f"\nDatos históricos para {dimension} - {domain}:")
        for a, v in zip(historical_years, historical_values):
            print(f"   {a}: {v:.2f}%")
        
        # 2. Codificar
        domain_encoded = self.le_domain.transform([domain])[0]
        dimension_encoded = self.le_dimension.transform([dimension])[0]
        
        # 3. Escalar valores históricos (normalización manual)
        mean_historical = np.mean(historical_values)
        std_historical = np.std(historical_values) if np.std(historical_values) > 0 else 1
        valores_norm = (historical_values - mean_historical) / std_historical
        
        # 4. Recursividad
        current_year = historical_values[-1]
        current_val_norm = valores_norm[-1]
        lag1 = valores_norm[-2] if len(valores_norm) >= 2 else current_val_norm
        lag2 = valores_norm[-3] if len(valores_norm) >= 3 else current_val_norm
        lag3 = valores_norm[-4] if len(valores_norm) >= 4 else current_val_norm
        
        # Calcular tendencia
        trend = current_val_norm - lag1
        
        # 5. Predicciones
        predicciones = []
        
        for i in range(years):
            year = begin_year + i
            
            # Construir features
            features_dict = {
                'year': year,
                'year_squared': year ** 2,
                'years_since_2010': year - 2010,
                'lag1': current_val_norm,
                'lag2': lag1,
                'lag3': lag2,
                'ma2': (current_val_norm + lag1) / 2 if len(valores_norm) >= 2 else current_val_norm,
                'ma3': (current_val_norm + lag1 + lag2) / 3 if len(valores_norm) >= 3 else current_val_norm,
                'trend': trend,
                'global_mean': current_val_norm,  # Aproximación
                'deviation_from_global': 0,
                'domain_mean': current_val_norm,
                'domain_std': std_historical,
                'domain_norm': current_val_norm,
                'sum_dimensions': 100,  # Aproximación
                'domain_encoded': domain_encoded,
                'dimension_encoded': dimension_encoded
            }
            
            features = [features_dict[col] for col in self.feature_cols]
            features_scaled = self.scaler.transform([features])
            
            pred_norm = self.model.predict(features_scaled)[0]
            
            # Desescalar
            pred_real = pred_norm * std_historical + mean_historical
            pred_real = max(pred_real, 0)
            pred_real = min(pred_real, 100)
            
            predicciones.append({
                'year': year,
                'dimension': dimension,
                'domain': domain,
                'percentage_contribution': float(pred_real)
            })
            
            # Actualizar recursividad
            lag3 = lag2
            lag2 = lag1
            lag1 = current_val_norm
            current_val_norm = pred_norm
            current_year = year
            trend = current_val_norm - lag1
        
        return pd.DataFrame(predicciones)
    
    def predict_all_dimensions(self, domain: str, begin_year: int, years: int):
        """Predice todas las dimensiones para un dominio."""
        results = {}
        
        for dimension in self.dimensions:
            try:
                data_frame = self.predict(dimension, domain, begin_year, years)
                results[dimension] = data_frame
                print(f" {dimension}: predicciones generadas")
            except Exception as e:
                print(f" {dimension}: {e}")
        
        if not results:
            return None
        
        # Crear tabla comparativa
        comparison = pd.DataFrame()
        comparison['year'] = list(range(begin_year, begin_year + years))
        
        for dimension, df in results.items():
            comparison[dimension] = df['percentage_contribution'].values
        
        # Verificar que la suma sea ~100%
        comparison['Suma'] = comparison[self.dimensions].sum(axis=1)
        
        return comparison



def predict_dimension(dimension: str, domain: str, begin_year: int , years: int ):
    predictor = DimensionContributionPredictor()
    return predictor.predict(dimension, domain, begin_year, years)


def predict_all_dimensions(domain: str, begin_year: int = 2026, years: int = 5):
    predictor = DimensionContributionPredictor()
    return predictor.predict_all_dimensions(domain, begin_year, years)


# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":

    print(" PREDICCIONES - CONTRIBUCIONES POR DIMENSIÓN")
    
    predictor = DimensionContributionPredictor()
    
    # Mostrar opciones disponibles
    predictor.show_options()
    
    # Ejemplo 1: Predicción de una dimensión específica
    print("\n 1. PREDICCIÓN: Educación - Nacional")
    print("-" * 40)
    df = predictor.predict('Educación', 'Nacional', 2026, 5)
    print("\n RESULTADOS:")
    print(df.to_string(index=False))
    
    print("\n 2. TODAS LAS DIMENSIONES - Nacional")
    print("-" * 40)
    comparison = predictor.predict_all_dimensions('Nacional', 2026, 5)
    
    if comparison is not None:
        print("\n TABLA COMPARATIVA:")
        print(comparison.round(2).to_string(index=False))
        
        # Guardar
        comparison.to_csv('predicciones_dimensiones_nacional.csv', index=False)
        print("\n Guardado en 'predicciones_dimensiones_nacional.csv'")