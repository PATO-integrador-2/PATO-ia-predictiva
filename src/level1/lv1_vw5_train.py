"""
Entrenamiento
Incidencia de Pobreza Multidimensioanl según sexo de la persona 
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

# MEJORES HIPERPARÁMETROS ENCONTRADOS

PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

# Prepara los datos para el entrenamiento.
def prepare_data():

    # Conectar a la base de datos y obtener la vista
    database = Database()
    data_frame = database.get_data_view_incidence_by_person_sex()
    data_frame = data_frame.dropna().copy()

    rename_map = {}
    for col in data_frame.columns:
        lower_col = col.lower()
        if lower_col == 'porcentaje':
            rename_map[col] = 'percentage'
        elif lower_col == 'dominio':
            rename_map[col] = 'domain'
        elif lower_col == 'anio':
            rename_map[col] = 'year'
        elif lower_col == 'sexo':
            rename_map[col] = 'sex'

    if rename_map:
        data_frame = data_frame.rename(columns=rename_map)

    # MOSTRAR INFORMACIÓN DE LOS DATOS

    print(f"   Registros totales: {len(data_frame)}")
    print(f"   Años: {data_frame['year'].min()} - {data_frame['year'].max()}")
    print(f"   Dominios: {data_frame['domain'].nunique()}")
    print(f"   Sexos: {data_frame['sex'].nunique()}")

    # CODIFICAR VARIABLES CATEGÓRICAS
    le_domain = LabelEncoder()
    le_sex = LabelEncoder()

    data_frame['domain_encoded'] = le_domain.fit_transform(data_frame['domain'])
    data_frame['sex_encoded'] = le_sex.fit_transform(data_frame['sex'])

    # CREAR FEATURES TEMPORALES

    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010

    # Lags por dominio y sexo
    data_frame['lag1'] = data_frame.groupby(['domain', 'sex'])['percentage'].shift(1)
    data_frame['lag2'] = data_frame.groupby(['domain', 'sex'])['percentage'].shift(2)
    data_frame['lag3'] = data_frame.groupby(['domain', 'sex'])['percentage'].shift(3)

    # Medias móviles
    data_frame['ma2'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: x.rolling(2, min_periods=1).mean().shift(1)
    )
    data_frame['ma3'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # Tendencia
    data_frame['trend'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )

    # FEATURES GLOBALES

    data_frame['global_mean'] = data_frame.groupby(['year', 'sex'])['percentage'].transform('mean')
    data_frame['deviation_from_global'] = data_frame['percentage'] - data_frame['global_mean']


    # FEATURES DE ESTRUCTURA POR DOMINIO Y SEXO

    data_frame['domain_mean'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform('mean')
    data_frame['domain_std'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform('std')
    data_frame['domain_norm'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # LIMPIEZA FINAL
    data_frame = data_frame.dropna()

    # FEATURES FINALES
    feature_cols = [
        'year',
        'year_squared',
        'years_since_2010',
        'lag1',
        'lag2',
        'lag3',
        'ma2',
        'ma3',
        'trend',
        'global_mean',
        'deviation_from_global',
        'domain_mean',
        'domain_std',
        'domain_norm',
        'domain_encoded',
        'sex_encoded'
    ]

    X = data_frame[feature_cols]
    y = data_frame['percentage']

    # ESCALAR FEATURES

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DIVISIÓN TEMPORAL

    years = sorted(data_frame['year'].unique())
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]

    train_mask = data_frame['year'].isin(train_years)
    test_mask = data_frame['year'].isin(test_years)

    # RETORNAR DATOS PREPARADOS

    return {
        'X_train': X_scaled[train_mask],
        'X_test': X_scaled[test_mask],
        'y_train': y[train_mask],
        'y_test': y[test_mask],
        'feature_cols': feature_cols,
        'le_domain': le_domain,
        'le_sex': le_sex,
        'scaler': scaler,
        'n_samples': len(data_frame),
        'train_years': train_years,
        'test_years': test_years,
        'sexes': sorted(data_frame['sex'].unique().tolist()),
        'domains': sorted(data_frame['domain'].unique().tolist())
    }

def train_lv1_vw5():
    """
    Entrenamiento para incidencia por sexo de la persona.
    """
    
    # 1. Cargar datos
    database = Database()
    data_frame = database.get_data_view_incidence_by_person_sex()
    
    # Limpiar datos de prueba
    data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)]
    
    print(f"Registros totales: {len(data_frame)}")
    print(f"Años: {data_frame['anio'].min()} - {data_frame['anio'].max()}")
    print(f"Dominios: {data_frame['dominio'].nunique()}")
    print(f"Sexos: {data_frame['sexo'].nunique()}")
    
    # 2. Verificar si hay suficientes años
    years_available = sorted(data_frame['anio'].unique())
    print(f"Años disponibles: {years_available}")
    
    if len(years_available) < 3:
        print("⚠️ ADVERTENCIA: Pocos años de datos disponibles")
        print("   El modelo tendrá bajo rendimiento")
    
    # 3. Preparar features
    # Codificar variables categóricas
    le_dominio = LabelEncoder()
    le_sexo = LabelEncoder()
    
    data_frame['dominio_encoded'] = le_dominio.fit_transform(data_frame['dominio'])
    data_frame['sexo_encoded'] = le_sexo.fit_transform(data_frame['sexo'])
    
    # Crear features de tendencia por dominio y sexo
    data_frame = data_frame.sort_values(['dominio', 'sexo', 'anio'])
    
    data_frame['lag1'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].shift(1)
    data_frame['lag2'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].shift(2)
    data_frame['lag3'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].shift(3)
    
    # Estadísticas del dominio
    data_frame['domain_mean'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].transform('mean')
    data_frame['domain_std'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].transform('std')
    data_frame['domain_norm'] = (data_frame['porcentaje'] - data_frame['domain_mean']) / data_frame['domain_std'].replace(0, 1)
    
    # Medias móviles
    data_frame['ma2'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].transform(
        lambda x: x.rolling(2, min_periods=1).mean()
    )
    data_frame['ma3'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # Tendencia
    data_frame['trend'] = data_frame.groupby(['dominio', 'sexo'])['porcentaje'].transform(
        lambda x: x.diff().fillna(0)
    )
    
    # Desviación del promedio global
    global_mean = data_frame['porcentaje'].mean()
    data_frame['deviation_from_global'] = data_frame['porcentaje'] - global_mean
    
    # Eliminar filas con NaN (primeros años)
    data_frame = data_frame.dropna()
    
    print(f"Registros después de features: {len(data_frame)}")
    
    # 4. Features
    feature_cols = [
        'dominio_encoded',
        'sexo_encoded',
        'lag1', 'lag2', 'lag3',
        'domain_mean', 'domain_std', 'domain_norm',
        'ma2', 'ma3',
        'trend',
        'deviation_from_global'
    ]
    
    X = data_frame[feature_cols]
    y = data_frame['porcentaje']
    
    # 5. Split temporal - USAR TODOS LOS AÑOS POSIBLES
    years = sorted(data_frame['anio'].unique())
    
    # Opción A: Split 80/20 por años
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    # Opción B: Si hay menos de 3 años, usar leave-one-year-out
    if len(years) < 3:
        print("⚠️ Pocos años disponibles. Usando leave-one-year-out")
        # Usar el año más reciente para prueba
        train_years = years[:-1]
        test_years = [years[-1]]
    
    train_mask = data_frame['anio'].isin(train_years)
    test_mask = data_frame['anio'].isin(test_years)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\nSplit temporal:")
    print(f"  Entrenamiento: {len(X_train)} registros (años {train_years})")
    print(f"  Prueba: {len(X_test)} registros (años {test_years})")
    
    # 6. Entrenar modelo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 7. Evaluar
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEVALUACIÓN:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # 8. Guardar modelo
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'models', 'train_lv1_vw5')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(le_dominio, os.path.join(model_dir, 'label_encoder_domain.pkl'))
    joblib.dump(le_sexo, os.path.join(model_dir, 'label_encoder_sex.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    # Guardar métricas
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    metrics = {
        'model': 'RandomForestRegressor_IncidenceBySex',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_importance': importance.to_dict('records'),
        'feature_names': feature_cols,
        'n_estimators': 100,
        'max_depth': 8,
        'train_date': datetime.now().isoformat(),
        'n_samples': len(data_frame),
        'train_years': [int(y) for y in train_years],
        'test_years': [int(y) for y in test_years],
        'domains': data_frame['dominio'].unique().tolist(),
        'sexes': data_frame['sexo'].unique().tolist()
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\nModelo guardado en: {model_dir}")
    print("ENTRENAMIENTO COMPLETADO")
    
    return model, metrics


if __name__ == "__main__":
    train_lv1_vw5()