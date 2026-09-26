"""
Entrenamiento para predicción de pobreza multidimensional por departamento.
Nivel 2 - Vista 1: Hogares por departamento
Incluye features temporales para predecir tendencias.
"""

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database

#Entrena el modelo de pobreza multidimensional por departamento con features temporales.
def train_lv2_vw1():
    
    # 1. Cargar datos
    database = Database()
    data_frame = database.get_data_view_household_poverty_by_department()
    
    print("ENTRENAMIENTO LV2_VW1 - POBREZA MULTIDIMENSIONAL POR DEPARTAMENTO")
    print(f"Registros totales: {len(data_frame)}")
    print(f"Años: {data_frame['anio'].min()} - {data_frame['anio'].max()}")
    print(f"Regiones: {data_frame['region'].nunique()}")
    print(f"Departamentos: {data_frame['departamento'].nunique()}")
    
    # 2. Limpiar datos
    data_frame = data_frame.dropna().copy()
    print(f"\nRegistros despues de limpiar: {len(data_frame)}")
    
    # 3. Convertir region y departamento a string
    data_frame['region'] = data_frame['region'].astype(str)
    data_frame['departamento'] = data_frame['departamento'].astype(str)
    
    # 4. Codificar variables categoricas
    le_region = LabelEncoder()
    le_departamento = LabelEncoder()
    
    data_frame['region_encoded'] = le_region.fit_transform(data_frame['region'])
    data_frame['departamento_encoded'] = le_departamento.fit_transform(data_frame['departamento'])
    
    # 5. Crear features temporales
    print("\nCreando features temporales...")
    
    data_frame['year'] = data_frame['anio']
    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010
    
    # Ordenar por region, departamento y año para calcular lags
    data_frame = data_frame.sort_values(['region', 'departamento', 'anio'])
    
    # Lags de IPM por region-departamento
    data_frame['ipm_lag1'] = data_frame.groupby(['region', 'departamento'])['ipm'].shift(1)
    data_frame['ipm_lag2'] = data_frame.groupby(['region', 'departamento'])['ipm'].shift(2)
    data_frame['ipm_lag3'] = data_frame.groupby(['region', 'departamento'])['ipm'].shift(3)
    
    # Tendencia del IPM
    data_frame['ipm_trend'] = data_frame.groupby(['region', 'departamento'])['ipm'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )
    
    # 6. Definir features y target
    feature_cols = [
        # Features temporales
        'year',
        'year_squared',
        'years_since_2010',
        'ipm_lag1',
        'ipm_lag2',
        'ipm_lag3',
        'ipm_trend',
        # Privaciones
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
        # Variables codificadas
        'region_encoded',
        'departamento_encoded'
    ]
    
    # Eliminar filas con NaN (por los lags)
    data_frame = data_frame.dropna()
    
    X = data_frame[feature_cols]
    y = data_frame['ipm']
    
    print(f"\nFeatures: {len(feature_cols)} columnas")
    print(f"Target: IPM (rango: {y.min():.3f} - {y.max():.3f})")
    print(f"Registros despues de lags: {len(data_frame)}")
    
    # 7. Split temporal (80% entrenamiento, 20% prueba)
    years = sorted(data_frame['anio'].unique())
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    train_mask = data_frame['anio'].isin(train_years)
    test_mask = data_frame['anio'].isin(test_years)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\nSplit temporal:")
    print(f"  Entrenamiento: {len(X_train)} registros (años {train_years[0]}-{train_years[-1]})")
    print(f"  Prueba: {len(X_test)} registros (años {test_years[0]}-{test_years[-1]})")
    
    # 8. Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 9. Entrenar modelo con mejores hiperparametros
    print("\nEntrenando modelo Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 10. Evaluar modelo
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("EVALUACION DEL MODELO")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # 11. Importancia de variables
    print("IMPORTANCIA DE VARIABLES")
    
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance.to_string(index=False))
    
    print("\nTOP 10 VARIABLES MAS IMPORTANTES:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # 12. Guardar modelo y artefactos
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'models', 'train_lv2_vw1')
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nGuardando modelo en: {model_dir}")
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(le_region, os.path.join(model_dir, 'label_encoder_region.pkl'))
    joblib.dump(le_departamento, os.path.join(model_dir, 'label_encoder_departamento.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    # 13. Guardar metricas y detalles del entrenamiento
    metrics = {
        'model': 'RandomForestRegressor_LV2_VW1',
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_importance': importance.to_dict('records'),
        'feature_names': feature_cols,
        'train_date': datetime.now().isoformat(),
        'n_samples': int(len(data_frame)),
        'train_years': [int(y) for y in train_years],
        'test_years': [int(y) for y in test_years],
        'n_features': len(feature_cols),
        'regions': [str(r) for r in data_frame['region'].unique().tolist()],
        'departments': [str(d) for d in data_frame['departamento'].unique().tolist()]
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Modelo guardado en: {model_dir}")
    
    return model, metrics


if __name__ == "__main__":
    train_lv2_vw1()