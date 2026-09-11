
"""
Entrenamiento del modelo para porcentaje de privaciones
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database


def prepare_data():
    
    database = Database()
    data_frame = database.get_data_view_average_deprivations()
    data_frame = data_frame.dropna().copy()
    
    # Limpiar nombres
    rename_map = {}
    for col in data_frame.columns:
        if col.lower() == 'porcentaje':
            rename_map[col] = 'porcentaje'
        elif col.lower() == 'dominio':
            rename_map[col] = 'dominio'
        elif col.lower() == 'anio':
            rename_map[col] = 'anio'
    
    if rename_map:
        data_frame = data_frame.rename(columns=rename_map)
    
    data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)]
    
    # Codificar dominio
    le_domain = LabelEncoder()
    data_frame['domain_encoded'] = le_domain.fit_transform(data_frame['dominio'])
    
    # Crear features
    data_frame = data_frame.copy()
    data_frame['year'] = data_frame['anio']
    
    # Variables derivadas (features)
    
    # 1. deviation_from_global (la más importante)
    data_frame['global_mean'] = data_frame.groupby('year')['porcentaje'].transform('mean')
    data_frame['deviation_from_global'] = data_frame['porcentaje'] - data_frame['global_mean']
    
    # 2. domain_percentile
    data_frame['domain_percentile'] = data_frame.groupby('year')['porcentaje'].rank(pct=True)
    
    # 3. lag1
    data_frame['lag1'] = data_frame.groupby('dominio')['porcentaje'].shift(1)
    
    # 4. ma2 (media móvil 2 años)
    data_frame['ma2'] = data_frame.groupby('dominio')['porcentaje'].transform(
        lambda x: x.rolling(2, min_periods=1).mean().shift(1)
    )
    
    # 5. global_ma3 (media móvil global 3 años)
    data_frame['global_ma3'] = data_frame.groupby('year')['porcentaje'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # 6. ma3 (media móvil 3 años)
    data_frame['ma3'] = data_frame.groupby('dominio')['porcentaje'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    
    # 7. acceleration
    data_frame['acceleration'] = data_frame.groupby('dominio')['porcentaje'].transform(
        lambda x: x.diff().diff().fillna(0)
    )
    
    # 8. pos_relative_max
    data_frame['pos_relative_max'] = data_frame.groupby('dominio')['porcentaje'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() - x.min() > 0 else 0
    )
    
    # 9. domain_norm
    data_frame['domain_norm'] = data_frame.groupby('dominio')['porcentaje'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    feature_cols = [
        'deviation_from_global',
        'domain_percentile',
        'lag1',
        'ma2',
        'global_ma3',
        'ma3',
        'acceleration',
        'global_mean',
        'pos_relative_max',
        'domain_norm'
    ]
    
    # Eliminar filas con NaN
    df_clean = data_frame.dropna()
    
    X = df_clean[feature_cols]
    y = df_clean['porcentaje']
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split temporal
    years = sorted(df_clean['year'].unique())
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    train_mask = df_clean['year'].isin(train_years)
    test_mask = df_clean['year'].isin(test_years)
    
    return {
        'X_train': X_scaled[train_mask],
        'X_test': X_scaled[test_mask],
        'y_train': y[train_mask],
        'y_test': y[test_mask],
        'feature_cols': feature_cols,
        'le_domain': le_domain,
        'scaler': scaler,
        'n_samples': len(df_clean),
        'train_years': train_years,
        'test_years': test_years
    }

# Entrenar el modelo 
def train_lv1_vw3():
    print(" ENTRENAMIENTO PRIVACIONES PORCENTAJE")
    
    # Preparar datos
    data = prepare_data()
    
    print(f"\n Datos:")
    print(f"   Entrenamiento: {len(data['X_train'])} registros (años {data['train_years'][0]}-{data['train_years'][-1]})")
    print(f"   Prueba: {len(data['X_test'])} registros (años {data['test_years'][0]}-{data['test_years'][-1]})")
    print(f"   Features: {len(data['feature_cols'])}")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    # Evaluar
    y_pred = model.predict(data['X_test'])
    mae = mean_absolute_error(data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))
    r2 = r2_score(data['y_test'], y_pred)
    
    print(f"\n EVALUACIÓN:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    # Importancia
    importance = pd.DataFrame({
        'Feature': data['feature_cols'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n IMPORTANCIA DE VARIABLES:")
    for _, row in importance.iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")
    
    # Guardar
    print("\n Guardando modelo")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'models', 'train_lv1_vw3')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(data['le_domain'], os.path.join(model_dir, 'label_encoder_domain.pkl'))
    joblib.dump(data['feature_cols'], os.path.join(model_dir, 'feature_cols.pkl'))
    
    metrics = {
        'model': 'RandomForestRegressor_Simple',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_importance': importance.to_dict('records'),
        'feature_names': data['feature_cols'],
        'train_date': datetime.now().isoformat(),
        'n_samples': data['n_samples']
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "-" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("-" * 60)
    print(f"\n Modelo guardado en: {model_dir}")
    
    return model, metrics


if __name__ == "__main__":
    train_lv1_vw3()