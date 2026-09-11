"""
Entrenamiento optimizado para incidencia de pobreza multidimensional por grupo de edad y area.
Usa los mejores hiperparametros encontrados.
Version con menos lags y promedio de años anteriores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

from database import Database

# Mejores hiperparametros encontrados
PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

# Prepara los datos para el entrenamiento.
def prepare_data():
    
    database = Database()
    data_frame = database.get_data_view_dashboard03_poverty_by_age()
    data_frame = data_frame.dropna().copy()
    
    rename_map = {
        'anio': 'year',
        'area_geografica': 'area',
        'pais': 'country',
        'grupo_erario': 'age_range',
        'valor_porcentaje': 'value'
    }
    
    data_frame = data_frame.rename(columns=rename_map)
    
    print(f"Registros totales: {len(data_frame)}")
    print(f"Años: {data_frame['year'].min()} - {data_frame['year'].max()}")
    print(f"Areas: {data_frame['area'].nunique()}")
    print(f"Paises: {data_frame['country'].nunique()}")
    print(f"Rangos de edad: {data_frame['age_range'].nunique()}")
    
    le_area = LabelEncoder()
    le_country = LabelEncoder()
    le_age_range = LabelEncoder()
    
    data_frame['area_encoded'] = le_area.fit_transform(data_frame['area'])
    data_frame['country_encoded'] = le_country.fit_transform(data_frame['country'])
    data_frame['age_range_encoded'] = le_age_range.fit_transform(data_frame['age_range'])
    
    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010
    

    # SOLO 2 LAGS (en lugar de 3)
    data_frame['value_lag1'] = data_frame.groupby(['country', 'area', 'age_range'])['value'].shift(1)
    data_frame['value_lag2'] = data_frame.groupby(['country', 'area', 'age_range'])['value'].shift(2)
    
    # NUEVA FEATURE: PROMEDIO DE AÑOS ANTERIORES

    data_frame['avg_previous_2'] = data_frame.groupby(['country', 'area', 'age_range'])['value'].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean()
    )
    data_frame['avg_previous_3'] = data_frame.groupby(['country', 'area', 'age_range'])['value'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # TENDENCIA (mantenemos)
    data_frame['value_trend'] = data_frame.groupby(['country', 'area', 'age_range'])['value'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )
    
    # FEATURES GLOBALES
    data_frame['global_mean'] = data_frame.groupby(['year', 'age_range'])['value'].transform('mean')
    data_frame['deviation_from_global'] = data_frame['value'] - data_frame['global_mean']
    
    data_frame = data_frame.dropna()
    
    # NUEVAS FEATURES
    feature_cols = [
        'year',
        'year_squared',
        'years_since_2010',
        'value_lag1',
        'value_lag2',
        'avg_previous_2',
        'avg_previous_3',
        'value_trend',
        'global_mean',
        'deviation_from_global',
        'area_encoded',
        'country_encoded',
        'age_range_encoded'
    ]
    
    x_matrix = data_frame[feature_cols]
    y_matrix = data_frame['value']
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_matrix)
    
    years = sorted(data_frame['year'].unique())
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    train_mask = data_frame['year'].isin(train_years)
    test_mask = data_frame['year'].isin(test_years)
    
    return {
        'x_train': x_scaled[train_mask],
        'x_test': x_scaled[test_mask],
        'y_train': y_matrix[train_mask],
        'y_test': y_matrix[test_mask],
        'feature_cols': feature_cols,
        'le_area': le_area,
        'le_country': le_country,
        'le_age_range': le_age_range,
        'scaler': scaler,
        'n_samples': len(data_frame),
        'train_years': train_years,
        'test_years': test_years,
        'areas': sorted(data_frame['area'].unique().tolist()),
        'countries': sorted(data_frame['country'].unique().tolist()),
        'age_ranges': sorted(data_frame['age_range'].unique().tolist())
    }

# Entrena el modelo con los mejores hiperparametros.
def train_lv3_vw3():

    print("ENTRENAMIENTO OPTIMIZADO V2")
    print("POBREZA MULTIDIMENSIONAL POR EDAD Y AREA")
    print("")
    
    data = prepare_data()
    
    print("")
    print("Datos preparados:")
    print(f"Entrenamiento: {len(data['x_train'])} registros")
    print(f"Prueba: {len(data['x_test'])} registros")
    print(f"Features: {len(data['feature_cols'])}")
    print("")
    
    print("Usando mejores hiperparametros:")
    print(f"   n_estimators: {PARAMS['n_estimators']}")
    print(f"   max_depth: {PARAMS['max_depth']}")
    print(f"   min_samples_split: {PARAMS['min_samples_split']}")
    print(f"   min_samples_leaf: {PARAMS['min_samples_leaf']}")
    print("")
    
    model = RandomForestRegressor(**PARAMS)
    model.fit(data['x_train'], data['y_train'])
    
    y_pred = model.predict(data['x_test'])
    mae = mean_absolute_error(data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))
    r2 = r2_score(data['y_test'], y_pred)
    
    print("EVALUACION:")
    print(f"   R²:  {r2:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print("")
    
    importance = pd.DataFrame({
        'Feature': data['feature_cols'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("TOP 10 VARIABLES MAS IMPORTANTES:")
    for _, row in importance.head(10).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")
    print("")
    
    print("Guardando modelo...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'models', 'train_lv3_vw3')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(data['le_area'], os.path.join(model_dir, 'label_encoder_area.pkl'))
    joblib.dump(data['le_country'], os.path.join(model_dir, 'label_encoder_country.pkl'))
    joblib.dump(data['le_age_range'], os.path.join(model_dir, 'label_encoder_age_range.pkl'))
    joblib.dump(data['feature_cols'], os.path.join(model_dir, 'feature_cols.pkl'))
    joblib.dump(data['areas'], os.path.join(model_dir, 'areas.pkl'))
    joblib.dump(data['countries'], os.path.join(model_dir, 'countries.pkl'))
    joblib.dump(data['age_ranges'], os.path.join(model_dir, 'age_ranges.pkl'))
    
    metrics = {
        'model': 'RandomForestRegressor_PovertyByAge',
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'params': PARAMS,
        'feature_importance': importance.to_dict('records'),
        'feature_names': data['feature_cols'],
        'areas': data['areas'],
        'countries': data['countries'],
        'age_ranges': data['age_ranges'],
        'train_date': datetime.now().isoformat(),
        'n_samples': data['n_samples']
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as file:
        json.dump(metrics, file, indent=2)
    
    print("")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Modelo guardado en: {model_dir}")


if __name__ == "__main__":
    train_lv3_vw3()