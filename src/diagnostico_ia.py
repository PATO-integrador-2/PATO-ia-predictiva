"""
Script de diagnostico para la IA predictiva.
Identifica problemas de dependencias, DLLs y configuracion.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_python_version():
    print_header("1. VERSION DE PYTHON")
    print(f"Python: {sys.version}")
    print(f"Ejecutable: {sys.executable}")
    print(f"Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    if sys.version_info.major == 3 and sys.version_info.minor >= 14:
        print("\n⚠️  ADVERTENCIA: Python 3.14 es muy nuevo.")
        print("   Muchas librerias no son compatibles.")
    elif sys.version_info.major == 3 and sys.version_info.minor >= 11:
        print("\n✅ Version de Python compatible.")
    else:
        print("\n⚠️  Version de Python antigua.")


def check_venv():
    print_header("2. ENTORNO VIRTUAL")
    in_venv = sys.prefix != sys.base_prefix
    print(f"En entorno virtual: {in_venv}")
    print(f"sys.prefix: {sys.prefix}")
    print(f"sys.base_prefix: {sys.base_prefix}")
    
    if in_venv:
        print("\n✅ Ejecutando dentro de un entorno virtual.")
    else:
        print("\n⚠️  NO estas en un entorno virtual.")


def check_dependencies():
    print_header("3. DEPENDENCIAS INSTALADAS")
    
    dependencies = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'joblib',
        'sqlalchemy', 'psycopg2', 'fastapi', 'uvicorn',
        'pydantic', 'dotenv', 'matplotlib', 'seaborn', 'plotly'
    ]
    
    installed = []
    missing = []
    broken = []
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'desconocida')
            installed.append((dep, version))
            print(f"   ✅ {dep}: {version}")
        except ImportError as e:
            missing.append(dep)
            print(f"   ❌ {dep}: NO INSTALADO ({e})")
        except Exception as e:
            broken.append((dep, str(e)))
            print(f"   ⚠️  {dep}: ERROR ({e})")
    
    print(f"\n📊 Resumen: {len(installed)} instalados, {len(missing)} faltantes, {len(broken)} rotos")
    
    return installed, missing, broken


def check_scipy_dll():
    print_header("4. VERIFICACION DE SCI PY (DLL)")
    print("Este es el problema principal segun el error.")
    
    try:
        import scipy
        print(f"   ✅ scipy importado: {scipy.__version__}")
    except ImportError as e:
        print(f"   ❌ ERROR importando scipy: {e}")
        print("\n   El problema es una DLL bloqueada por Windows Defender.")
        return False
    except Exception as e:
        print(f"   ❌ ERROR desconocido: {e}")
        return False
    
    try:
        from scipy.optimize import curve_fit
        print(f"   ✅ scipy.optimize.curve_fit importado correctamente")
    except ImportError as e:
        print(f"   ❌ ERROR importando curve_fit: {e}")
        return False
    
    return True


def check_windows_defender():
    print_header("5. VERIFICACION DE WINDOWS DEFENDER")
    
    if platform.system() != 'Windows':
        print("   No es Windows. Saltando verificacion.")
        return
    
    print("   Verificando politicas de Control de Aplicaciones...")
    
    try:
        # Verificar si Smart App Control esta activo
        result = subprocess.run(
            ['powershell', '-Command', 
             'Get-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\CI\\Policy" -ErrorAction SilentlyContinue | Select-Object -Property *'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.stdout:
            print(f"\n   Politicas de CI encontradas:")
            print(result.stdout)
        else:
            print("   No se encontraron politicas de CI.")
    except Exception as e:
        print(f"   No se pudo verificar: {e}")
    
    print("\n   💡 Para solucionar el error de DLL:")
    print("   1. Abre Seguridad de Windows")
    print("   2. Ve a 'Control de aplicaciones y navegador'")
    print("   3. Ve a 'Proteccion basada en reputacion'")
    print("   4. Desactiva temporalmente 'Bloquear aplicaciones potencialmente no deseadas'")
    print("   5. Vuelve a ejecutar el script")
    print("   6. Vuelve a activar la proteccion")


def check_paths():
    print_header("6. VERIFICACION DE RUTAS")
    
    base_dir = Path(__file__).resolve().parent
    
    paths_to_check = [
        ('src', base_dir / 'src'),
        ('src/level1', base_dir / 'src' / 'level1'),
        ('src/level2', base_dir / 'src' / 'level2'),
        ('src/level3', base_dir / 'src' / 'level3'),
        ('src/api', base_dir / 'src' / 'api'),
        ('models', base_dir / 'models'),
        ('models/train_lv1_vw1', base_dir / 'models' / 'train_lv1_vw1'),
        ('models/train_lv1_vw2', base_dir / 'models' / 'train_lv1_vw2'),
        ('models/train_lv2_vw1', base_dir / 'models' / 'train_lv2_vw1'),
        ('models/train_lv3_vw1', base_dir / 'models' / 'train_lv3_vw1'),
        ('models/train_lv3_vw2', base_dir / 'models' / 'train_lv3_vw2'),
    ]
    
    for name, path in paths_to_check:
        if path.exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: NO EXISTE ({path})")


def check_model_files():
    print_header("7. VERIFICACION DE ARCHIVOS DE MODELOS")
    
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'
    
    if not models_dir.exists():
        print(f"   ❌ La carpeta 'models' no existe: {models_dir}")
        return
    
    for model_folder in models_dir.iterdir():
        if model_folder.is_dir():
            print(f"\n   📁 {model_folder.name}/")
            required_files = ['model.pkl', 'scaler.pkl']
            for file in required_files:
                file_path = model_folder / file
                if file_path.exists():
                    size = file_path.stat().st_size / 1024
                    print(f"      ✅ {file} ({size:.1f} KB)")
                else:
                    print(f"      ❌ {file} NO EXISTE")


def check_database_connection():
    print_header("8. VERIFICACION DE CONEXION A BASE DE DATOS")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT')
        database = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Database: {database}")
        print(f"   User: {user}")
        
        if not all([host, port, database, user]):
            print("\n   ⚠️  Faltan variables de entorno. Verifica el archivo .env")
            return
        
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=host, port=port, dbname=database,
                user=user, password=os.getenv('DB_PASSWORD'),
                connect_timeout=5
            )
            print("\n   ✅ Conexion a BD exitosa")
            conn.close()
        except Exception as e:
            print(f"\n   ❌ Error de conexion a BD: {e}")
            print("   Verifica que Tailscale este activo.")
    except ImportError:
        print("   ⚠️  python-dotenv no instalado. Saltando verificacion.")


def generate_report():
    print_header("RESUMEN DEL DIAGNOSTICO")
    
    print("""
    POSIBLES PROBLEMAS IDENTIFICADOS:
    
    1. DLL de scipy bloqueada por Windows Defender
       -> Solucion: Desactivar temporalmente Smart App Control
    
    2. Python 3.14 es muy nuevo
       -> Solucion: Usar Python 3.11 o 3.12
    
    3. Dependencias rotas
       -> Solucion: Reinstalar con pip install --force-reinstall
    
    4. Entorno virtual mal configurado
       -> Solucion: Recrear el entorno virtual
    
    PLAN DE ACCION:
    
    PASO 1: Verificar Windows Defender
    PASO 2: Reinstalar scipy
    PASO 3: Si falla, recrear entorno virtual
    PASO 4: Si falla, usar Python 3.11
    """)


def main():
    print("\n" + "=" * 80)
    print("  DIAGNOSTICO DE IA PREDICTIVA")
    print("  Proyecto: PATO-ia-predictiva")
    print("=" * 80)
    
    check_python_version()
    check_venv()
    check_dependencies()
    check_scipy_dll()
    check_windows_defender()
    check_paths()
    check_model_files()
    check_database_connection()
    generate_report()
    
    print("\n" + "=" * 80)
    print("  FIN DEL DIAGNOSTICO")
    print("=" * 80)


if __name__ == "__main__":
    main()