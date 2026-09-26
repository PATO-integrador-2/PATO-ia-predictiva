"""
Prediccion de pobreza multidimensional por departamento.
Nivel 2 - Vista 1: Hogares por departamento
Incluye proyeccion con ajuste de curva para las privaciones.
Incluye agrupacion por año para obtener series temporales reales.
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

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from database import Database


# FUNCIONES DE PROYECCION

def linear(x, a, b):
    return a + b * x


def quadratic(x, a, b, c):
    return a + b * x + c * x**2


def logarithmic(x, a, b):
    return a + b * np.log(x + 1)


def exponential(x, a, b):
    return a * np.exp(b * x)

#Proyecta una privacion usando el mejor modelo de ajuste.
def project_privation(historical_values, future_years):

    n = len(historical_values)
    x_data = np.arange(n)
    y_data = np.array(historical_values)
    
    models = {
        'linear': linear,
        'quadratic': quadratic,
        'logarithmic': logarithmic,
        'exponential': exponential
    }
    
    best_r2 = -float('inf')
    best_function = None
    best_params = None
    best_name = None
    
    for name, function in models.items():
        try:
            params, _ = curve_fit(function, x_data, y_data, maxfev=5000)
            y_pred = function(x_data, *params)
            r2 = r2_score(y_data, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_function = function
                best_params = params
                best_name = name
        except:
            pass
    
    # Si no se pudo ajustar ningun modelo, usar promedio
    if best_function is None:
        return np.array([np.mean(y_data)] * future_years)
    
    # Proyectar años futuros
    x_future = np.arange(n, n + future_years)
    y_future = best_function(x_future, *best_params)
    
    # Asegurar que este entre 0 y 1
    y_future = np.clip(y_future, 0, 1)
    
    return y_future


# CLASE PREDICTOR

#Predictor para pobreza multidimensional por departamento.
class DepartmentPovertyPredictor:

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, 'models', 'train_lv2_vw1')
        
        print("POBREZA MULTIDIMENSIONAL POR DEPARTAMENTO")
        
        required_files = ['model.pkl', 'scaler.pkl', 'label_encoder_region.pkl', 
            'label_encoder_departamento.pkl', 'feature_cols.pkl']
        
        for file in required_files:
            file_path = os.path.join(self.model_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No se encontro: {file_path}")
        
        # Cargar artefactos
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_region = joblib.load(os.path.join(self.model_dir, 'label_encoder_region.pkl'))
        self.le_departamento = joblib.load(os.path.join(self.model_dir, 'label_encoder_departamento.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        
        # Cargar metricas
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r', encoding='utf-8') as f:
            self.metrics = json.load(f)
        
        print(f"\nModelo cargado correctamente")
        print(f"  R²: {self.metrics['r2']:.4f}")
        print(f"  MAE: {self.metrics['mae']:.4f}")
        print(f"  RMSE: {self.metrics['rmse']:.4f}")
        print(f"  Features: {len(self.feature_cols)}")
        print("=" * 80)
        
        print("\nREGIONES DISPONIBLES:")
        for region in self.le_region.classes_:
            print(f"   - {region}")
        
        print("\nDEPARTAMENTOS DISPONIBLES:")
        for depto in self.le_departamento.classes_:
            print(f"   - {depto}")
        
        print("=" * 80)
    
    # Agrupa los datos por año y calcula el promedio de cada variable
    def _aggregate_by_year(self, data_frame):
        privacion_cols = [
            'personas_que_habitan_ese_hogar',
            'privacion_bajo_logro_educativo',
            'privacion_analfabetismo',
            'privacion_inasistencia_escolar',
            'privacion_rezago_escolar',
            'privacion_atencion_integral_primera_infancia',
            'privacion_trabajo_infantil',
            'privacion_no_aseguramiento_salud',
            'privacion_barreras_acceso_salud',
            'privacion_desempleo_larga_duracion',
            'privacion_tasa_empleo_formal',
            'privacion_no_acceso_agua_mejorada',
            'privacion_inadecuada_eliminacion_excretas',
            'privacion_inadecuado_material_pisos',
            'privacion_inadecuado_material_paredes',
            'privacion_hacinamiento_critico',
            'ipm'
        ]
        
        # Agrupar por año y calcular promedio
        aggregated = data_frame.groupby('anio')[privacion_cols].mean().reset_index()
        
        return aggregated
    
    # Obtiene datos historicos agregados por año para una region y departamento.
    def _get_historical_data(self, region, departamento, years_hist=5):

        database = Database()
        data_frame = database.get_data_view_household_poverty_by_department()

        data_frame['region'] = data_frame['region'].astype(str)
        data_frame['departamento'] = data_frame['departamento'].astype(str)

        data_frame = data_frame[
            (data_frame['region'] == str(region)) & 
            (data_frame['departamento'] == str(departamento))
        ]
        
        # Agrupar por año
        data_frame = self._aggregate_by_year(data_frame)
        data_frame = data_frame.sort_values('anio')

        if len(data_frame) < years_hist:
            raise ValueError(
                f"Datos insuficientes para {region} - {departamento}. "
                f"Se encontraron {len(data_frame)} años, se necesitan {years_hist}"
            )

        return data_frame.tail(years_hist)
    
    # Prepara las features para el modelo.
    def _prepare_features(self, row):

        features_dict = {}
        for col in self.feature_cols:
            if col == 'region_encoded':
                region_value = str(row['region'])
                features_dict[col] = self.le_region.transform([region_value])[0]
            elif col == 'departamento_encoded':
                depto_value = str(row['departamento'])
                features_dict[col] = self.le_departamento.transform([depto_value])[0]
            else:
                features_dict[col] = row[col]
        
        features = [features_dict[col] for col in self.feature_cols]
        features_scaled = self.scaler.transform([features])
        
        return features_scaled
    
    # Predice el IPM de un hogar especifico.
    def predict_household(self, row):
        features_scaled = self._prepare_features(row)
        prediction = self.model.predict(features_scaled)[0]
        return float(prediction)
    
    # Predice el IPM para departamentos en multiples años.
    def predict_department_trend(self, region, departamentos, año_inicio=2026, años=5):

        results = []
        
        for depto in departamentos:
            try:
                historical = self._get_historical_data(region, depto, years_hist=5)
                
                privacion_cols = [
                    'personas_que_habitan_ese_hogar',
                    'privacion_bajo_logro_educativo',
                    'privacion_analfabetismo',
                    'privacion_inasistencia_escolar',
                    'privacion_rezago_escolar',
                    'privacion_atencion_integral_primera_infancia',
                    'privacion_trabajo_infantil',
                    'privacion_no_aseguramiento_salud',
                    'privacion_barreras_acceso_salud',
                    'privacion_desempleo_larga_duracion',
                    'privacion_tasa_empleo_formal',
                    'privacion_no_acceso_agua_mejorada',
                    'privacion_inadecuada_eliminacion_excretas',
                    'privacion_inadecuado_material_pisos',
                    'privacion_inadecuado_material_paredes',
                    'privacion_hacinamiento_critico'
                ]
                
                ipm_historico = historical['ipm'].values
                ipm_real = ipm_historico[-1]
                anio_real = historical['anio'].iloc[-1]
                
                # PROYECTAR CADA PRIVACION CON AJUSTE DE CURVA
                proyecciones = {}
                for col in privacion_cols:
                    if col == 'personas_que_habitan_ese_hogar':
                        # Mantener constante
                        proyecciones[col] = [historical[col].mean()] * años
                    else:
                        valores = historical[col].values
                        proyecciones[col] = project_privation(valores, años)
                
                # Inicializar lags
                lag1 = ipm_historico[-1] if len(ipm_historico) >= 1 else 0
                lag2 = ipm_historico[-2] if len(ipm_historico) >= 2 else lag1
                lag3 = ipm_historico[-3] if len(ipm_historico) >= 3 else lag2
                trend = lag1 - lag2
                
                base_row = {
                    'region': region,
                    'departamento': depto,
                }
                
                for i in range(años):
                    year = año_inicio + i
                    base_row['anio'] = year
                    
                    # Features temporales
                    base_row['year'] = year
                    base_row['year_squared'] = year ** 2
                    base_row['years_since_2010'] = year - 2010
                    
                    base_row['ipm_lag1'] = lag1
                    base_row['ipm_lag2'] = lag2
                    base_row['ipm_lag3'] = lag3
                    base_row['ipm_trend'] = trend
                    
                    # Usar valores proyectados
                    for col in privacion_cols:
                        base_row[col] = proyecciones[col][i]
                    
                    ipm_pred = self.predict_household(base_row)
                    
                    results.append({
                        'departamento': depto,
                        'region': region,
                        'anio_real': int(anio_real),
                        'ipm_real': float(ipm_real),
                        'año': year,
                        'ipm_predicho': ipm_pred,
                        'cambio': ipm_pred - ipm_real,
                        'cambio_porcentaje': ((ipm_pred - ipm_real) / ipm_real * 100) if ipm_real > 0 else 0
                    })
                    
                    # Actualizar lags
                    lag3 = lag2
                    lag2 = lag1
                    lag1 = ipm_pred
                    trend = lag1 - lag2
                
            except Exception as e:
                print(f"  Error con {depto}: {e}")
        
        return pd.DataFrame(results)
    
    # Predice el IPM de un hogar especifico con sus caracteristicas.
    def predict_household_poverty(self, region, departamento, household_data):

        row = {
            'region': region,
            'departamento': departamento,
        }
        row.update(household_data)
        
        # Agregar features temporales por defecto
        row['year'] = 2025
        row['year_squared'] = 2025 ** 2
        row['years_since_2010'] = 2025 - 2010
        
        # Si no tiene lags, usar 0
        if 'ipm_lag1' not in row:
            row['ipm_lag1'] = 0
        if 'ipm_lag2' not in row:
            row['ipm_lag2'] = 0
        if 'ipm_lag3' not in row:
            row['ipm_lag3'] = 0
        if 'ipm_trend' not in row:
            row['ipm_trend'] = 0
        
        return self.predict_household(row)



# EJECUCION PRINCIPAL


if __name__ == "__main__":
    print("PREDICCIONES - POBREZA MULTIDIMENSIONAL POR DEPARTAMENTO")
    print("")
    
    predictor = DepartmentPovertyPredictor()
    
    # 1. PREDICCION DE TENDENCIA POR DEPARTAMENTOS

    print("")
    print("1. PREDICCION DE TENDENCIA POR DEPARTAMENTOS")
    print("")

    region = "Caribe"
    departamentos = ["Atlántico", "Bolívar", "Magdalena"]
    
    df_trend = predictor.predict_department_trend(region, departamentos, año_inicio=2026, años=5)
    
    if not df_trend.empty:
        print(f"  Region: {region}")
        print(f"  Año inicio: 2026")
        print(f"  Años: 5")
        print("")
        print("  RESULTADOS:")
        print(df_trend[['departamento', 'año', 'ipm_predicho', 'cambio_porcentaje']].to_string(index=False))
    
    print("")
    print("PREDICCIONES COMPLETADAS")