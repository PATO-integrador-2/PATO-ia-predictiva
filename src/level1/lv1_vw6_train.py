"""
Entrenamiento 
Incidencia de Pobreza Multidimensional según sexo del jefe del hogar
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

PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
} 

#  Prepara los datos para el entrenamiento.
def prepare_data():

    database = Database()
    data_frame = database.get_data_view_incidence_by_household_head_sex()
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

    print(f"   Registros totales: {len(data_frame)}")
    print(f"   Años: {data_frame['year'].min()} - {data_frame['year'].max()}")
    print(f"   Dominios: {data_frame['domain'].nunique()}")
    print(f"   Sexos: {data_frame['sex'].nunique()}")

    le_domain = LabelEncoder()
    le_sex = LabelEncoder()

    data_frame['domain_encoded'] = le_domain.fit_transform(data_frame['domain'])
    data_frame['sex_encoded'] = le_sex.fit_transform(data_frame['sex'])

    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010

    data_frame['lag1'] = data_frame.groupby(['domain', 'sex'])['percentage'].shift(1)
    data_frame['lag2'] = data_frame.groupby(['domain', 'sex'])['percentage'].shift(2)
    data_frame['lag3'] = data_frame.groupby(['domain', 'sex'])['percentage'].shift(3)

    data_frame['ma2'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: x.rolling(2, min_periods=1).mean().shift(1)
    )
    data_frame['ma3'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    data_frame['trend'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )

    data_frame['global_mean'] = data_frame.groupby(['year', 'sex'])['percentage'].transform('mean')
    data_frame['deviation_from_global'] = data_frame['percentage'] - data_frame['global_mean']

    data_frame['domain_mean'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform('mean')
    data_frame['domain_std'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform('std')
    data_frame['domain_norm'] = data_frame.groupby(['domain', 'sex'])['percentage'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

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
        'domain_encoded',
        'sex_encoded'
    ]

    X = data_frame[feature_cols]
    y = data_frame['percentage']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
        'le_sex': le_sex,
        'scaler': scaler,
        'n_samples': len(data_frame),
        'train_years': train_years,
        'test_years': test_years,
        'sexes': sorted(data_frame['sex'].unique().tolist()),
        'domains': sorted(data_frame['domain'].unique().tolist())
    }

#Entrena el modelo de incidencia de pobreza por sexo del jefe del hogar.

def train_lv1_vw6():

    print("ENTRENANDO MODELO - INCIDENCIA POR SEXO DEL JEFE DEL HOGAR")
    print("")

    data = prepare_data()

    print("")
    print("Datos:")
    print(f"   Entrenamiento: {len(data['X_train'])} registros (años {data['train_years'][0]}-{data['train_years'][-1]})")
    print(f"   Prueba: {len(data['X_test'])} registros (años {data['test_years'][0]}-{data['test_years'][-1]})")
    print(f"   Features: {len(data['feature_cols'])}")

    print("")
    print("Usando hiperparametros:")
    print(f"   n_estimators: {PARAMS['n_estimators']}")
    print(f"   max_depth: {PARAMS['max_depth']}")
    print(f"   min_samples_split: {PARAMS['min_samples_split']}")
    print(f"   min_samples_leaf: {PARAMS['min_samples_leaf']}")

    model = RandomForestRegressor(**PARAMS)
    model.fit(data['X_train'], data['y_train'])

    y_pred = model.predict(data['X_test'])
    mae = mean_absolute_error(data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))
    r2 = r2_score(data['y_test'], y_pred)

    print("")
    print("EVALUACION:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")

    importance = pd.DataFrame({
        'Feature': data['feature_cols'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("")
    print("TOP 10 VARIABLES MAS IMPORTANTES:")
    for _, row in importance.head(10).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")

    print("")
    print("Guardando modelo...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'models', 'train_lv1_vw6')
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(data['scaler'], os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(data['le_domain'], os.path.join(model_dir, 'label_encoder_domain.pkl'))
    joblib.dump(data['le_sex'], os.path.join(model_dir, 'label_encoder_sex.pkl'))
    joblib.dump(data['feature_cols'], os.path.join(model_dir, 'feature_cols.pkl'))
    joblib.dump(data['sexes'], os.path.join(model_dir, 'sexes.pkl'))
    joblib.dump(data['domains'], os.path.join(model_dir, 'domains.pkl'))

    metrics = {
        'model': 'RandomForestRegressor_IncidenceByHeadSex',
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'params': PARAMS,
        'feature_importance': importance.to_dict('records'),
        'feature_names': data['feature_cols'],
        'sexes': data['sexes'],
        'domains': data['domains'],
        'train_date': datetime.now().isoformat(),
        'n_samples': data['n_samples']
    }

    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("")
    print("ENTRENAMIENTO COMPLETADO")
    print("")
    print(f"Modelo guardado en: {model_dir}")

    return model, metrics


if __name__ == "__main__":
    train_lv1_vw6()