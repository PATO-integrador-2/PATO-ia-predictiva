
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
from datetime import datetime 
from database import Database

def train_lv1_vw1():

    # 1. Cargar datos
    database = Database()
    data_frame = database.get_data_view_ipm_domain()
    
    # 2. Limpiar datos de prueba
    data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)]
    
    print(f"Datos cargados: {len(data_frame)} registros")
    print(f"Dominios: {data_frame['dominio'].nunique()}")
    print(f"Años: {data_frame['anio'].min()} - {data_frame['anio'].max()}")

    # 3. Crear features mejorados
    data_frame = data_frame.sort_values(['dominio', 'anio'])
    
    # Features básicas
    data_frame['year'] = data_frame['anio']
    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010
    
    # Features de tendencia por dominio
    data_frame['ipm_lag1'] = data_frame.groupby('dominio')['ipm'].shift(1)
    data_frame['ipm_lag2'] = data_frame.groupby('dominio')['ipm'].shift(2)
    data_frame['domain_trend'] = data_frame.groupby('dominio')['ipm'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )
    
    # Codificar dominio
    le = LabelEncoder()
    data_frame['domain_encoded'] = le.fit_transform(data_frame['dominio'])
    
    # 4. Eliminar filas con NaN (primeros años)
    data_frame = data_frame.dropna()
    
    print(f"\nDatos después de features: {len(data_frame)} registros")
    
    # 5. Seleccionar features y target
    feature_cols = [
        'year',
        'year_squared',
        'years_since_2010',
        'ipm_lag1',
        'ipm_lag2',
        'domain_trend',
        'domain_encoded'
    ]
    
    X = data_frame[feature_cols]
    y = data_frame['ipm']
    
    print(f"\nFeatures: {feature_cols}")
    print(f"Registros de entrenamiento: {len(X)}")
    
    # 6. Split temporal
    train_mask = data_frame['year'] <= 2020
    test_mask = data_frame['year'] >= 2021
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\nSplit temporal:")
    print(f"Entrenamiento: {len(X_train)} registros (hasta 2020)")
    print(f"Prueba: {len(X_test)} registros (2021-2025)")
    
    # 7. Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 8. Entrenar
    print("\nEntrenando Random Forest")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 9. Evaluar
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEVALUACIÓN:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # 10. Importancia de variables
    print("\nIMPORTANCIA DE VARIABLES:")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance.to_string(index=False))
    
    # 11. Guardar modelo y artefactos
    print("\nModelo guardado")
    os.makedirs('models/train_lv1_vw1', exist_ok=True)
    
    joblib.dump(model, 'models/train_lv1_vw1/model.pkl')
    joblib.dump(scaler, 'models/train_lv1_vw1/scaler.pkl')
    joblib.dump(le, 'models/train_lv1_vw1/label_encoder.pkl')
    joblib.dump(feature_cols, 'models/train_lv1_vw1/feature_cols.pkl')

    # Guardar métricas y detalles del entrenamiento
    metrics = {
        'model': 'RandomForestRegressor_Global_Improved',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_importance': importance.to_dict('records'),
        'feature_names': feature_cols,
        'n_estimators': 150,
        'max_depth': 8,
        'train_date': datetime.now().isoformat(),
        'n_samples': len(data_frame),
        'years': [int(y) for y in data_frame['year'].unique()]
    }
    
    with open('models/train_lv1_vw1/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Modelo guardado en models/train_lv1_vw1/")

    print("ENTRENAMIENTO COMPLETADO")
    
    return model, metrics


if __name__ == "__main__":
    train_lv1_vw1()