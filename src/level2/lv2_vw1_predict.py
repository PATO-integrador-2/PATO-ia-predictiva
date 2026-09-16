"""
Prediccion de pobreza multidimensional por departamento.
Nivel 2 - Vista 1: Hogares por departamento
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
                raise FileNotFoundError(f"No se encontró: {file_path}")
        
        print(f"\nModelo de IPM region-Departamento cargado")
        self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.le_region = joblib.load(os.path.join(self.model_dir, 'label_encoder_region.pkl'))
        self.le_departamento = joblib.load(os.path.join(self.model_dir, 'label_encoder_departamento.pkl'))
        self.feature_cols = joblib.load(os.path.join(self.model_dir, 'feature_cols.pkl'))
        
        # Cargar metricas
        with open(os.path.join(self.model_dir, 'metrics.json'), 'r', encoding='utf-8') as f:
            self.metrics = json.load(f)
        
        # ==========================================
        # MOSTRAR OPCIONES DISPONIBLES
        # ==========================================
        print(f"\nModelo cargado correctamente")
        print(f"  R²: {self.metrics['r2']:.4f}")
        print(f"  MAE: {self.metrics['mae']:.4f}")
        print(f"  RMSE: {self.metrics['rmse']:.4f}")
        print(f"  Features: {len(self.feature_cols)}")
        print("=" * 80)
        
        print("\nFEATURES DEL MODELO:")
        for col in self.feature_cols:
            print(f"   - {col}")
        
        print("\nREGIONES DISPONIBLES:")
        for region in self.le_region.classes_:
            print(f"   - {region}")
        
        print("\nDEPARTAMENTOS DISPONIBLES:")
        for depto in self.le_departamento.classes_:
            print(f"   - {depto}")
        
        print("=" * 80)
    
    def _get_historical_data(self, region, departamento, years_hist: int = 5):
        database = Database()
        data_frame = database.get_data_view_household_poverty_by_department()

        data_frame['region'] = data_frame['region'].astype(str)
        data_frame['departamento'] = data_frame['departamento'].astype(str)
    
        data_frame = data_frame[
            (data_frame['region'] == str(region)) & 
            (data_frame['departamento'] == str(departamento))
        ]
    
        data_frame = data_frame.sort_values('anio')
    
        if len(data_frame) < years_hist:
            raise ValueError(
                f"Datos insuficientes para {region} - {departamento}. "
                f"Se encontraron {len(data_frame)} años, se necesitan {years_hist}"
            )
    
        return data_frame.tail(years_hist)
    
    def _prepare_features(self, row):
        features_dict = {}
        for col in self.feature_cols:
            if col == 'region_encoded':
                region_value = str(row['region'])
                try:
                    features_dict[col] = self.le_region.transform([region_value])[0]
                except ValueError:
                    raise ValueError(
                        f"Región '{region_value}' no reconocida. "
                        f"Regiones válidas: {list(self.le_region.classes_)}"
                    )
            elif col == 'departamento_encoded':
                depto_value = str(row['departamento'])
                try:
                    features_dict[col] = self.le_departamento.transform([depto_value])[0]
                except ValueError:
                    raise ValueError(
                        f"Departamento '{depto_value}' no reconocido. "
                        f"Departamentos válidos: {list(self.le_departamento.classes_)}"
                    )
            else:
                features_dict[col] = row[col]
        
        features = [features_dict[col] for col in self.feature_cols]
        features_scaled = self.scaler.transform([features])
        
        return features_scaled
    
    def predict_household(self, row):
        features_scaled = self._prepare_features(row)
        prediction = self.model.predict(features_scaled)[0]
        return float(prediction)
    
    def predict_households(self, data_frame):
        results = data_frame.copy()
        
        features_list = []
        for _, row in results.iterrows():
            features_scaled = self._prepare_features(row)
            features_list.append(features_scaled[0])
        
        predictions = self.model.predict(features_list)
        results['ipm_predicted'] = predictions
        
        return results
    
    def predict_department_trend(self, region: str, departamentos: list, año_inicio: int = 2026, años: int = 5):
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
                
                # Inicializar lags
                lag1 = ipm_historico[-1] if len(ipm_historico) >= 1 else 0
                lag2 = ipm_historico[-2] if len(ipm_historico) >= 2 else lag1
                lag3 = ipm_historico[-3] if len(ipm_historico) >= 3 else lag2
                trend = lag1 - lag2
                
                total_privations = historical[privacion_cols].sum(axis=1).mean()
                
                base_row = {
                    'region': region,
                    'departamento': depto,
                    'total_privations': total_privations
                }
                
                for col in privacion_cols:
                    base_row[col] = historical[col].mean()
                
                for i in range(años):
                    year = año_inicio + i
                    base_row['anio'] = year
                    
                    base_row['year'] = year
                    base_row['year_squared'] = year ** 2
                    base_row['years_since_2010'] = year - 2010
                    
                    base_row['ipm_lag1'] = lag1
                    base_row['ipm_lag2'] = lag2
                    base_row['ipm_lag3'] = lag3
                    base_row['ipm_trend'] = trend
                    
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
                    
                    lag3 = lag2
                    lag2 = lag1
                    lag1 = ipm_pred
                    trend = lag1 - lag2   
                
            except Exception as e:
                print(f"  Error con {depto}: {e}")
        
        return pd.DataFrame(results)


# EJECUCION PRINCIPAL

if __name__ == "__main__":
    print("PREDICCIONES - POBREZA MULTIDIMENSIONAL POR DEPARTAMENTO")
    print("")
    
    predictor = DepartmentPovertyPredictor()

    # 1. PREDICCION DE UN HOGAR
    print("")
    print("1. PREDICCION DE UN HOGAR")
    print("")
    
    hogar = {
    'region': 1,
    'departamento': 8,
    'personas_que_habitan_ese_hogar': 6,          
    'privacion_bajo_logro_educativo': 1,
    'privacion_analfabetismo': 1,
    'privacion_inasistencia_escolar': 0,
    'privacion_rezago_escolar': 1,
    'privacion_atencion_integral_primera_infancia': 0,
    'privacion_trabajo_infantil': 0,
    'privacion_no_aseguramiento_salud': 1,
    'privacion_barreras_acceso_salud': 0,
    'privacion_desempleo_larga_duracion': 1,
    'privacion_tasa_empleo_formal': 1,
    'privacion_no_acceso_agua_mejorada': 0,
    'privacion_inadecuada_eliminacion_excretas': 0,
    'privacion_inadecuado_material_pisos': 1,
    'privacion_inadecuado_material_paredes': 0,
    'privacion_hacinamiento_critico': 1
}
    
    try:
        ipm_hogar = predictor.predict_household(hogar)
        print(f"  Region: {hogar['region']}")
        print(f"  Departamento: {hogar['departamento']}")
        print(f"  Personas: {hogar['personas_que_habitan_ese_hogar']}")
        print(f"  IPM Predicho: {ipm_hogar:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 2. PREDICCION DE TENDENCIA POR DEPARTAMENTOS
    print("")
    print("2. PREDICCION DE TENDENCIA POR DEPARTAMENTOS")
    print("")

    region = 1
    departamentos = [8, 13, 47]
    
    df_trend = predictor.predict_department_trend(region, departamentos, año_inicio=2026, años=3)
    
    if not df_trend.empty:
        print(f"  Region: {region}")
        print(f"  Año inicio: 2026")
        print(f"  Años: 3")
        print("")
        print("  RESULTADOS:")
        print(df_trend[['departamento', 'año', 'ipm_predicho', 'cambio_porcentaje']].to_string(index=False))
    
    print("")
    print("PREDICCIONES COMPLETADAS")