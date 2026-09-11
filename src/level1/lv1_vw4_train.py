"""
Entrenamiento
Contribuciones a la incidencia Ajustada del IPM
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

# Best hyperparameters found
PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}


def prepare_data():
    """Prepare data for training."""
    
    database = Database()
    data_frame = database.get_data_view_dimension_contribution()
    data_frame = data_frame.dropna().copy()
    
    # Renombrar columnas en ingles
    rename_map = {}
    for col in data_frame.columns:
        if col.lower() == 'porcentaje':
            rename_map[col] = 'percentage'
        elif col.lower() == 'dominio':
            rename_map[col] = 'domain'
        elif col.lower() == 'anio':
            rename_map[col] = 'year'
        elif col.lower() == 'dimension':
            rename_map[col] = 'dimension'
    
    if rename_map:
        data_frame = data_frame.rename(columns=rename_map)
    
    print(f"   Filas Totales: {len(data_frame)}")
    print(f"   Años: {data_frame['year'].min()} - {data_frame['year'].max()}")
    print(f"   Dominios: {data_frame['domain'].nunique()}")
    print(f"   Dimensiones: {data_frame['dimension'].nunique()}")
    
    # Codificar variables categóricas
    le_domain = LabelEncoder()
    le_dimension = LabelEncoder()
    
    data_frame['domain_encoded'] = le_domain.fit_transform(data_frame['domain'])
    data_frame['dimension_encoded'] = le_dimension.fit_transform(data_frame['dimension'])
    
    # Crar variables
    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010
    
    # Lags por dominio y dimensión
    data_frame['lag1'] = data_frame.groupby(['domain', 'dimension'])['percentage'].shift(1)
    data_frame['lag2'] = data_frame.groupby(['domain', 'dimension'])['percentage'].shift(2)
    data_frame['lag3'] = data_frame.groupby(['domain', 'dimension'])['percentage'].shift(3)
    
    # Moviendo promedio
    data_frame['ma2'] = data_frame.groupby(['domain', 'dimension'])['percentage'].transform(
        lambda x: x.rolling(2, min_periods=1).mean().shift(1)
    )
    data_frame['ma3'] = data_frame.groupby(['domain', 'dimension'])['percentage'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    
    # Trend
    data_frame['trend'] = data_frame.groupby(['domain', 'dimension'])['percentage'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )
    
    # Caracteristicas globales (por año y dimensión)
    data_frame['global_mean'] = data_frame.groupby(['year', 'dimension'])['percentage'].transform('mean')
    data_frame['deviation_from_global'] = data_frame['percentage'] - data_frame['global_mean']
    
    # Caracteristicas estructurales
    data_frame['domain_mean'] = data_frame.groupby(['domain', 'dimension'])['percentage'].transform('mean')
    data_frame['domain_std'] = data_frame.groupby(['domain', 'dimension'])['percentage'].transform('std')
    data_frame['domain_norm'] = data_frame.groupby(['domain', 'dimension'])['percentage'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Caracteristicas de la estructura
    data_frame['sum_dimensions'] = data_frame.groupby(['year', 'domain'])['percentage'].transform('sum')
    
    data_frame = data_frame.dropna()
    
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
        'sum_dimensions',
        'domain_encoded',
        'dimension_encoded'
    ]
    
    X = data_frame[feature_cols]
    y = data_frame['percentage']
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split Temporal
    years = sorted(data_frame['year'].unique())
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    train_mask = data_frame['year'].isin(train_years)
    test_mask = data_frame['year'].isin(test_years)
    
    return {
        'X_train': X_scaled[train_mask],
        'X_test': X_scaled[test_mask],
        'y_train': y[train_mask],
        'y_test': y[test_mask],
        'feature_cols': feature_cols,
        'le_domain': le_domain,
        'le_dimension': le_dimension,
        'scaler': scaler,
        'n_samples': len(data_frame),
        'train_years': train_years,
        'test_years': test_years,
        'dimensions': sorted(data_frame['dimension'].unique().tolist()),
        'domains': sorted(data_frame['domain'].unique().tolist())
    }


def train_lv1_vw4():

    print("ENTRENAMIENTO DEL MODELO DE CONTRIBUCIÓN POR DIMENSIÓN")

    data = prepare_data()
    
    print(f"\nDatos:")
    print(f"Entrenamiento: {len(data['X_train'])} records (years {data['train_years'][0]}-{data['train_years'][-1]})")
    print(f"Pruebas: {len(data['X_test'])} records (years {data['test_years'][0]}-{data['test_years'][-1]})")
    print(f"Características: {len(data['feature_cols'])}")
    
    model = RandomForestRegressor(**PARAMS)
    model.fit(data['X_train'], data['y_train'])
    
    # Evaluation
    y_pred = model.predict(data['X_test'])
    mae = mean_absolute_error(data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))
    r2 = r2_score(data['y_test'], y_pred)
    
    print(f"\nEVALUACIÓN:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Importancia de Caracteristicas
    importance = pd.DataFrame({
        'Feature': data['feature_cols'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
    for _, row in importance.head(10).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")
    
    # Save model
    print("\nMODELO GUARDADO...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'models', 'train_lv1_vw4')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(data['le_domain'], os.path.join(model_dir, 'label_encoder_domain.pkl'))
    joblib.dump(data['le_dimension'], os.path.join(model_dir, 'label_encoder_dimension.pkl'))
    joblib.dump(data['feature_cols'], os.path.join(model_dir, 'feature_cols.pkl'))
    joblib.dump(data['dimensions'], os.path.join(model_dir, 'dimensions.pkl'))
    joblib.dump(data['domains'], os.path.join(model_dir, 'domains.pkl'))
    
    metrics = {
        'model': 'RandomForestRegressor_DimensionContribution',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'params': PARAMS,
        'feature_importance': importance.to_dict('records'),
        'feature_names': data['feature_cols'],
        'dimensions': data['dimensions'],
        'domains': data['domains'],
        'train_date': datetime.now().isoformat(),
        'n_samples': data['n_samples']
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("ENTRENAMIENTO COMPLETADO")
    print(f"\nModelo guardado en: {model_dir}")
    
    return model, metrics


if __name__ == "__main__":
    train_lv1_vw4()