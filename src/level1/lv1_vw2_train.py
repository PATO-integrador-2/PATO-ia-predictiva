"""
Entrenamiento para IPM por privaciones y dominio.
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


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database


# ==========================================
# REENTRENAMIENTO CON FEATURES TEMPORALES
# ==========================================

def retrain_with_temporal_features():
    """
    Reentrena el modelo CON features temporales para que las predicciones cambien año a año.
    """
    
    print("REENTRENAMIENTO CON FEATURES TEMPORALES")
    print("=" * 80)
    
    database = Database()
    data_frame = database.get_data_view_household_poverty_by_department()
    data_frame = data_frame.dropna().copy()
    
    # Convertir a string
    data_frame['region'] = data_frame['region'].astype(str)
    data_frame['departamento'] = data_frame['departamento'].astype(str)
    
    print(f"Registros totales: {len(data_frame)}")
    print(f"Años: {data_frame['anio'].min()} - {data_frame['anio'].max()}")
    
    # Codificar
    le_region = LabelEncoder()
    le_departamento = LabelEncoder()
    data_frame['region_encoded'] = le_region.fit_transform(data_frame['region'])
    data_frame['departamento_encoded'] = le_departamento.fit_transform(data_frame['departamento'])
    
    # ==========================================
    # FEATURES TEMPORALES
    # ==========================================
    data_frame['year'] = data_frame['anio']
    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010
    
    data_frame = data_frame.sort_values(['region', 'departamento', 'anio'])
    
    # Lags de IPM
    data_frame['ipm_lag1'] = data_frame.groupby(['region', 'departamento'])['ipm'].shift(1)
    data_frame['ipm_lag2'] = data_frame.groupby(['region', 'departamento'])['ipm'].shift(2)
    data_frame['ipm_lag3'] = data_frame.groupby(['region', 'departamento'])['ipm'].shift(3)
    
    # Tendencia
    data_frame['ipm_trend'] = data_frame.groupby(['region', 'departamento'])['ipm'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )
    
    data_frame = data_frame.dropna()
    
    feature_cols = [
        'year',
        'year_squared',
        'years_since_2010',
        'ipm_lag1',
        'ipm_lag2',
        'ipm_lag3',
        'ipm_trend',
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
        'region_encoded',
        'departamento_encoded'
    ]
    
    x_matrix = data_frame[feature_cols]
    y_matrix = data_frame['ipm']
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_matrix)
    
    years = sorted(data_frame['year'].unique())
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    
    train_mask = data_frame['year'].isin(train_years)
    test_mask = data_frame['year'].isin(test_years)
    
    x_train = x_scaled[train_mask]
    x_test = x_scaled[test_mask]
    y_train = y_matrix[train_mask]
    y_test = y_matrix[test_mask]
    
    print(f"\nSplit temporal:")
    print(f"   Entrenamiento: {len(x_train)} registros (años {train_years[0]}-{train_years[-1]})")
    print(f"   Prueba: {len(x_test)} registros (años {test_years[0]}-{test_years[-1]})")
    print(f"   Features: {len(feature_cols)}")
    
    # ==========================================
    # ENTRENAR MODELO
    # ==========================================
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEVALUACIÓN:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTOP 10 VARIABLES MAS IMPORTANTES:")
    for _, row in importance.head(10).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")
    
    # ==========================================
    # GUARDAR MODELO REENTRENADO
    # ==========================================
    print("\nGuardando modelo reentrenado...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base_dir, 'models', 'train_lv2_vw1')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(le_region, os.path.join(model_dir, 'label_encoder_region.pkl'))
    joblib.dump(le_departamento, os.path.join(model_dir, 'label_encoder_departamento.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    joblib.dump(sorted(data_frame['region'].unique().tolist()), os.path.join(model_dir, 'regions.pkl'))
    joblib.dump(sorted(data_frame['departamento'].unique().tolist()), os.path.join(model_dir, 'departments.pkl'))
    
    metrics = {
        'model': 'RandomForestRegressor_DepartmentPoverty_v2',
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'best_params': params,
        'feature_importance': importance.to_dict('records'),
        'feature_names': feature_cols,
        'regions': sorted(data_frame['region'].unique().tolist()),
        'departments': sorted(data_frame['departamento'].unique().tolist()),
        'train_date': datetime.now().isoformat(),
        'n_samples': len(data_frame)
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as file:
        json.dump(metrics, file, indent=2)
    
    print(f"\nModelo guardado en: {model_dir}")
    print("=" * 80)


# ==========================================
# ENTRENAMIENTO ORIGINAL
# ==========================================

def train_lv1_vw2():

    # 1. Cargar datos
    database = Database()
    data_frame = database.get_data_view_deprivations_variable()

    # 2. Limpiar datos
    data_frame = data_frame.dropna().copy()

    # 2.5 Escalar datos por grupo
    def scale_by_group(df, group_cols, target_col):
        df_scaled = df.copy()
        scaler_dict = {}
    
        for (dominio, variable), group in df.groupby(group_cols):
            scaler = StandardScaler()
            df_scaled.loc[group.index, f'{target_col}_scaled'] = scaler.fit_transform(
                group[[target_col]].values
            ).flatten()
            scaler_dict[(dominio, variable)] = scaler
    
        return df_scaled, scaler_dict

    data_frame, scalers = scale_by_group(data_frame, ['dominio', 'variable'], 'ipm')

    print(f"Datos cargados: {len(data_frame)} registros")
    print(f"Dominios: {data_frame['dominio'].nunique()}")
    print(f"Variables: {data_frame['variable'].nunique()}")
    print(f"Años: {data_frame['anio'].min()} - {data_frame['anio'].max()}")
    print(f"Escaladores creados: {len(scalers)}")
    print(f"Datos escalados: {len(data_frame)} registros")
    print(f"Datos con IPM escalado: {len(data_frame)} registros")

    # 3. Preparar datos
    data_frame = data_frame[['dominio', 'anio', 'variable', 'ipm', 'ipm_scaled']].copy()

    labels_domains = LabelEncoder()
    labels_variables = LabelEncoder()
    data_frame['domain_encoded'] = labels_domains.fit_transform(data_frame['dominio'])
    data_frame['variable_encoded'] = labels_variables.fit_transform(data_frame['variable'])

    # 4. Crear features
    data_frame['year'] = data_frame['anio']
    data_frame['year_squared'] = data_frame['year'] ** 2
    data_frame['years_since_2010'] = data_frame['year'] - 2010

    data_frame['ipm_lag1'] = data_frame.groupby(['dominio', 'variable'])['ipm_scaled'].shift(1)
    data_frame['ipm_lag2'] = data_frame.groupby(['dominio', 'variable'])['ipm_scaled'].shift(2)
    data_frame['ipm_lag3'] = data_frame.groupby(['dominio', 'variable'])['ipm_scaled'].shift(3)
    data_frame['domain_trend'] = data_frame.groupby(['dominio', 'variable'])['ipm_scaled'].transform(
        lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
    )
    data_frame['variable_trend'] = data_frame.groupby(['dominio', 'variable'])['ipm_scaled'].transform(
    lambda x: x.diff().fillna(0).rolling(3, min_periods=1).mean()
)

    data_frame = data_frame.dropna()
    print(f"Datos después de crear features: {len(data_frame)} registros")

    features_cols = [
        'year',
        'year_squared',
        'years_since_2010',
        'ipm_lag1',
        'ipm_lag2',
        'ipm_lag3',
        'domain_trend',
        'variable_trend',
        'domain_encoded',
        'variable_encoded'
    ]

    x_axis = data_frame[features_cols]
    y_axis = data_frame['ipm_scaled']

    # 5. Split temporal (80/20)
    years = sorted(data_frame['year'].unique())
    split_idx = int(len(years) * 0.8)
    train_years = years[:split_idx]
    test_years = years[split_idx:]

    train_mask = data_frame['year'].isin(train_years)
    test_mask = data_frame['year'].isin(test_years)

    x_train = x_axis[train_mask]
    y_train = y_axis[train_mask]
    x_test = x_axis[test_mask]
    y_test = y_axis[test_mask]

    print(f"\nSplit temporal:")
    print(f"   - Entrenamiento: {len(x_train)} registros (años {train_years[0]}-{train_years[-1]})")
    print(f"   - Prueba: {len(x_test)} registros (años {test_years[0]}-{test_years[-1]})")

    # 6. Escalar y entrenar modelo global
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nEVALUACIÓN GLOBAL:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")

    # 7. Importancia de variables
    importance = pd.DataFrame({
        'Feature': features_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nIMPORTANCIA DE VARIABLES:")
    print(importance.to_string(index=False))

    print("\nTOP 10 VARIABLES MÁS IMPORTANTES:")
    for _, row in importance.head(10).iterrows():
        print(f"      - {row['Feature']}: {row['Importance']:.4f}")

    # 8. Entrenamiento de modelos por variable
    variables = data_frame['variable'].unique()
    models_by_variable = {}
    metrics_by_variable = {}

    for variable in variables:
        print(f"\nEntrenamiento de modelo para variable: {variable}")

        data_frame_var = data_frame[data_frame['variable'] == variable].copy()

        if len(data_frame_var) < 5:
            print(f"   Datos insuficientes ({len(data_frame_var)} años), saltando...")
            continue

        var_feature_cols = [
            'year',
            'year_squared',
            'years_since_2010',
            'ipm_lag1',
            'ipm_lag2',
            'ipm_lag3',
            'domain_trend',
            'variable_trend',
            'domain_encoded'
        ] 

        x_var = data_frame_var[var_feature_cols]
        y_var = data_frame_var['ipm_scaled']

        train_mask_var = data_frame_var['year'].isin(train_years)
        test_mask_var = data_frame_var['year'].isin(test_years)

        x_train_var = x_var[train_mask_var]
        x_test_var = x_var[test_mask_var]
        y_train_var = y_var[train_mask_var]
        y_test_var = y_var[test_mask_var]

        scaler_var = StandardScaler()
        x_train_var_scaled = scaler_var.fit_transform(x_train_var)
        x_test_var_scaled = scaler_var.transform(x_test_var)

        model_var = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        model_var.fit(x_train_var_scaled, y_train_var)

        y_pred_var = model_var.predict(x_test_var_scaled)
        mae_var = mean_absolute_error(y_test_var, y_pred_var)
        rmse_var = np.sqrt(mean_squared_error(y_test_var, y_pred_var))
        r2_var = r2_score(y_test_var, y_pred_var)

        models_by_variable[variable] = model_var
        metrics_by_variable[variable] = {
            'mae': float(mae_var),
            'rmse': float(rmse_var),
            'r2': float(r2_var)
        }

        print(f"   MAE: {mae_var:.4f}, RMSE: {rmse_var:.4f}, R²: {r2_var:.4f}")

    # 9. Guardar modelo y artefactos
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    global_dir = os.path.join(base_dir, 'models', 'train_lv1_vw2')
    by_variable_dir = os.path.join(base_dir, 'models', 'train_lv1_vw2', 'privations_by_variable')

    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(by_variable_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(global_dir, 'scaler_global.pkl'))
    joblib.dump(model, os.path.join(global_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(global_dir, 'scaler.pkl'))
    joblib.dump(labels_domains, os.path.join(global_dir, 'label_encoder_domain.pkl'))
    joblib.dump(labels_variables, os.path.join(global_dir, 'label_encoder_variable.pkl'))
    joblib.dump(features_cols, os.path.join(global_dir, 'feature_cols.pkl'))
    joblib.dump(scalers, os.path.join(global_dir, 'scalers_by_group.pkl'))
    joblib.dump(sorted(data_frame['dominio'].unique().tolist()), os.path.join(global_dir, 'domains.pkl'))
    joblib.dump(sorted(data_frame['variable'].unique().tolist()), os.path.join(global_dir, 'variables.pkl'))

    for variable, model_var in models_by_variable.items():
        var_name = variable.replace(" ", "_").replace('(', '').replace(')', '').lower()
        joblib.dump(model_var, os.path.join(by_variable_dir, f'model_{var_name}.pkl'))

    metrics = {
        'model': 'RandomForestRegressor_PrivationsOnly',
        'global': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'feature_importance': importance.to_dict('records'),
            'feature_names': features_cols
        },
        'by_variable': metrics_by_variable,
        'variables': list(variables),
        'domains': sorted(data_frame['dominio'].unique().tolist()),
        'n_estimators': 150,
        'max_depth': 10,
        'train_date': datetime.now().isoformat(),
        'n_samples': len(data_frame)
    }

    with open(os.path.join(global_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("ENTRENAMIENTO COMPLETADO")
    
    print(f"\nModelos guardados en:")
    print(f"   - Global: {global_dir}")
    print(f"   - Por variable: {by_variable_dir}")

    return model, metrics


if __name__ == "__main__":
    # Ejecutar primero el reentrenamiento con features temporales
    retrain_with_temporal_features()
    
    # Luego el entrenamiento original
    train_lv1_vw2()