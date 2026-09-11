# test_preprocessing.py
import sys
sys.path.append('src')
from database import Database
from preprocessing import DataPreprocessor

# 1. Cargar datos
db = Database()
df = db.get_data_view_ipm_domain()  # ← CAMBIAR AQUÍ
print(f"📊 Datos cargados: {len(df)} registros")
print(df.head())

# 2. Preprocesar
preprocessor = DataPreprocessor()
data = preprocessor.prepare_nivel1_data(df)

print("\n📈 Datos preparados:")
print(f"   x_train: {data['x_train'].shape}")
print(f"   x_test: {data['x_test'].shape}")
print(f"   features: {data['feature_names']}")
print(f"   target: {data['target_name']}")

# 3. Guardar preprocesadores
preprocessor.save_preprocessors()
print("\n Preprocesadores guardados")
print("\n Preprocesadores guardados")