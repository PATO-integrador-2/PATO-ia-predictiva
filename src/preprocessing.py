# Limpia y codifica los datos especificamente para el nivel 1 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Any 
import joblib 
import os 

# Preprocesamiento de datos para el modelo en general 
class DataPreprocessor:
    def __init__(self, target_column: str = "value"):
        self.target_column = target_column
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = []
    
    def prepare_level1_data (
        self, 
        data_frame: pd.DataFrame, 
        test_size: float= 0.2,
        temporal: bool = True
    ) -> Dict[str, Any]:

        data_frame = data_frame.copy()

        rename_map = {
            'anio': 'year',
            'dominio': 'domain',
            'ipm': 'ipm'
        }

        data_frame = data_frame.rename(columns=rename_map)

        if 'ipm' in data_frame.columns:
            self.target_column = 'ipm'

        if temporal: 
            data_frame = data_frame.sort_values('year')

        #col -> columnas

        categorical_cols = ['domain']
        for col in categorical_cols: 
            if col in data_frame.columns: 
                label_encoder = LabelEncoder()
                data_frame[f'{col}_encoded'] = label_encoder.fit_transform(data_frame[col].astype(str))
                self.label_encoders[col] = label_encoder
                data_frame = data_frame.drop(columns=[col])
    
        feature_cols = [col for col in data_frame.columns if col != self.target_column]
        self.feature_columns = feature_cols

        x_matrix = data_frame[feature_cols] 
        y_matrix = data_frame[self.target_column]

        if temporal:
            split_idx = int(len(x_matrix) * (1 - test_size))  
            x_train = x_matrix.iloc[:split_idx] 
            x_test = x_matrix.iloc[split_idx:]  
            y_train = y_matrix.iloc[:split_idx] 
            y_test = y_matrix.iloc[split_idx:]  
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_matrix, y_matrix, test_size=test_size, random_state=42
            )

        scaler = StandardScaler()  
        x_train_scaled = scaler.fit_transform(x_train)  
        x_test_scaled = scaler.transform(x_test)  
        self.scalers['level1'] = scaler  
        
        return {
            'x_train': x_train_scaled,
            'x_test': x_test_scaled,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': feature_cols,
            'x_train_original': x_train,
            'x_test_original': x_test,
            'target_name': self.target_column
        }

    def prepare_level2_data (
            self, 
            data_frame: pd.DataFrame, 
            test_size: float= 0.2,
            temporal: bool = True
        ) -> Dict[str, Any]:
    
            data_frame = data_frame.copy()
    
            rename_map = {
                'anio': 'year',
                'dominio': 'domain',
                'variable': 'variable',
                'ipm': 'ipm'
            }
    
            data_frame = data_frame.rename(columns=rename_map)
    
            if 'ipm' in data_frame.columns:
                self.target_column = 'ipm'
    
            if temporal: 
                data_frame = data_frame.sort_values('year')
    
            #col -> columnas
    
            categorical_cols = ['domain', 'variable']
            for col in categorical_cols: 
                if col in data_frame.columns: 
                    label_encoder = LabelEncoder()
                    data_frame[f'{col}_encoded'] = label_encoder.fit_transform(data_frame[col].astype(str))
                    self.label_encoders[col] = label_encoder
                    data_frame = data_frame.drop(columns=[col])
        
            feature_cols = [col for col in data_frame.columns if col != self.target_column]
            self.feature_columns = feature_cols
    
            x_matrix = data_frame[feature_cols] 
            y_matrix = data_frame[self.target_column]
    
            if temporal:
                split_idx = int(len(x_matrix) * (1 - test_size))  
                x_train = x_matrix.iloc[:split_idx] 
                x_test = x_matrix.iloc[split_idx:]  
                y_train = y_matrix.iloc[:split_idx] 
                y_test = y_matrix.iloc[split_idx:]  
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_matrix, y_matrix, test_size=test_size, random_state=42
                )
    
            scaler = StandardScaler()  
            x_train_scaled = scaler.fit_transform(x_train)  
            x_test_scaled = scaler.transform(x_test)  
            self.scalers['level2'] = scaler  
            
            return {
                'x_train': x_train_scaled,
                'x_test': x_test_scaled,
                'y_train': y_train.values,
                'y_test': y_test.values,
                'feature_names': feature_cols,
                'x_train_original': x_train,
                'x_test_original': x_test,
                'target_name': self.target_column
            }

    def save_preprocessors(self, path: str = 'models/preprocessors'):
        os.makedirs(path, exist_ok=True)  
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')  
        joblib.dump(self.scalers, f'{path}/scalers.pkl')  
        joblib.dump(self.feature_columns, f'{path}/feature_columns.pkl') 
    
    def load_preprocessors(self, path: str = 'models/preprocessors'):
        self.label_encoders = joblib.load(f'{path}/label_encoders.pkl')  
        self.scalers = joblib.load(f'{path}/scalers.pkl') 
        self.feature_columns = joblib.load(f'{path}/feature_columns.pkl') 

    # Falta la parte 3 
    