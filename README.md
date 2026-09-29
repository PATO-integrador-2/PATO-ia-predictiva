# PATO – IA Predictiva

Servicio de predicción para el proyecto PATO. Expone por HTTP (FastAPI) modelos de
Machine Learning que proyectan indicadores de **Pobreza Multidimensional (IPM)** hacia
el futuro, a partir de las vistas analíticas de una base de datos PostgreSQL.

- **Stack:** Python 3.12, FastAPI, scikit-learn (Random Forest), pandas, SQLAlchemy, reportlab.
- **Puerto por defecto:** `11010` (variable `PORT`).
- **Docs interactivas:** `http://localhost:11010/docs` (Swagger generado por FastAPI).

## Contenido

1. [Arquitectura](#arquitectura)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Modelos por nivel y vista](#modelos-por-nivel-y-vista)
4. [Configuración y ejecución local](#configuración-y-ejecución-local)
5. [Entrenamiento](#entrenamiento)
6. [API](#api)
7. [Docker y despliegue](#docker-y-despliegue)
8. [Notas y limitaciones conocidas](#notas-y-limitaciones-conocidas)

## Arquitectura

```
PostgreSQL (vistas vw_*)
        │  src/database.py  (SQLAlchemy, credenciales por .env)
        ▼
  *_train.py  ──►  models/train_<nivel>_<vista>/  (model.pkl, scaler, encoders, metrics.json)
                          │
                          ▼
  *_predict.py (clases Predictor)  ──►  src/api/main.py (FastAPI)  ──►  clientes / API gateway
```

- Cada vista tiene un script de **entrenamiento** y una clase **Predictor**.
- Los artefactos entrenados se versionan en `models/` y se copian a la imagen Docker.
- Al arrancar, `main.py` carga todos los predictores en el diccionario `predictors`.
  Si uno falla al cargar queda en `None`: la API sigue arriba y ese endpoint responde `500`.
- Los predictores de nivel 1 y 3 hacen **predicción recursiva**: leen los últimos años reales
  de la base de datos y cada año predicho alimenta los rezagos (`lag`) del siguiente.
  Por eso **la base de datos debe estar disponible también en tiempo de predicción**.

## Estructura del repositorio

```
src/
  database.py            Clase Database: conexión y una función por vista SQL
  preprocessing.py       DataPreprocessor (limpieza/codificación genérica, niveles 1 y 2)
  test_preprocessing.py  Script manual de prueba del preprocesamiento
  api/
    main.py              App FastAPI: modelos Pydantic, carga de predictores y endpoints
    pdf_export.py        generate_pdf_report(): exporta registros a PDF tabular (reportlab)
  level1/                lv1_vw{1..6}_train.py / _predict.py
  level2/                lv2_vw1_train.py / _predict.py
  level3/                lv3_vw{1..3}_train.py / _predict.py
models/                  Artefactos entrenados por vista + metrics.json
Endpoints nivel 1|2|3    Ejemplos de payloads por endpoint
dockerfile, .github/     Imagen y pipelines CI/CD
```

Convención de nombres: `lv<nivel>_vw<vista>`. Nivel 1 = nacional/dominios, nivel 2 = departamentos
y hogares, nivel 3 = Latinoamérica.

## Modelos por nivel y vista

Todos son `RandomForestRegressor`. Las métricas provienen de `models/*/metrics.json`
(conjunto de prueba temporal; en la vista 1 entrena hasta 2020 y evalúa de 2021 en adelante).

| Vista | Nivel | Qué predice | Vista SQL de origen | Predictor | R² | MAE |
|---|---|---|---|---|---|---|
| lv1_vw1 | 1 | Incidencia del IPM por dominio | `vw_ipm_by_domain` | `DomainPredictor` | 0.944 | 1.572 |
| lv1_vw2 | 1 | Privación por variable y dominio (un modelo global + uno por variable) | `vw_deprivations_by_variable` | `PrivationsPredictor` | 0.965 | 0.180 |
| lv1_vw3 | 1 | Porcentaje promedio de privaciones de la población pobre | `vw_average_deprivations` | `AverageSimplePredictor` | 0.942 | 0.227 |
| lv1_vw4 | 1 | Contribución por dimensión a la incidencia ajustada | `vw_dimension_contribution` | `DimensionContributionPredictor` | 0.996 | 0.598 |
| lv1_vw5 | 1 | Incidencia por sexo de la persona | `vw_incidence_by_person_sex` | `IncidenceBySexPredictor` | 0.996 | 0.241 |
| lv1_vw6 | 1 | Incidencia por sexo del jefe del hogar | `vw_incidence_by_household_head_sex` | `IncidenceByHeadSexPredictor` | 0.903 | 1.318 |
| lv2_vw1 | 2 | IPM por departamento y por hogar | `vw_household_poverty_by_department` | `DepartmentPovertyPredictor` | 0.999 | 0.0009 |
| lv3_vw1 | 3 | Contribución relativa de privaciones en LATAM | `vw_dashboard03_deprivation_contribution` | `ContributionLatamPredictor` | 0.919 | 0.672 |
| lv3_vw2 | 3 | Población en pobreza multidimensional (cifras nacionales) | `vw_dashboard03_national_poverty` | `NationalPovertyPredictor` | 0.993 | 0.950 |
| lv3_vw3 | 3 | Pobreza por grupo de edad y área | `vw_dashboard03_poverty_by_age` | `PovertyByAgePredictorV2` | 0.968 | 2.904 |

Detalles relevantes:

- **Features típicas:** `year`, `year²`, `years_since_2010`, rezagos (`lag1`, `lag2`…), medias móviles,
  tendencia por grupo, desviación frente al valor global y variables categóricas codificadas con `LabelEncoder`.
  Las variables se escalan con `StandardScaler`; el scaler y los encoders se guardan junto al modelo.
- **Nivel 2:** además del Random Forest, cada privación se proyecta al futuro con ajuste de curvas
  (lineal, cuadrática, logarítmica o exponencial; gana la de mejor R²) usando `scipy.optimize.curve_fit`.
- Los dominios de prueba (`__test__`) se excluyen al cargar la vista de IPM por dominio.

## Configuración y ejecución local

Requisitos: Python 3.12 y acceso a la base de datos PostgreSQL.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz (está en `.gitignore`):

```env
DB_HOST=...
DB_PORT=5432
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
```

Verifica la conexión y las 10 vistas:

```bash
python src/database.py
```

Levanta la API:

```bash
python src/api/main.py            # http://localhost:11010
```

Comprueba que los modelos cargaron con `GET /health`.

## Entrenamiento

Ejecuta los scripts **desde la raíz del repo**: guardan en rutas relativas `models/...`.

```bash
python src/level1/lv1_vw1_train.py
python src/level2/lv2_vw1_train.py
python src/level3/lv3_vw2_train.py
```

Cada script lee su vista, construye las features, hace split temporal, entrena, imprime MAE/RMSE/R²
y guarda `model.pkl`, `scaler.pkl`, encoders, `feature_cols.pkl` y `metrics.json` en
`models/train_<nivel>_<vista>/`. Tras reentrenar, haz commit de `models/` para que el cambio llegue a la imagen.

## API

Todas las predicciones son `POST` con JSON. `begin_year` (por defecto `2026`) y `years` (por defecto `5`)
son opcionales. La respuesta siempre tiene la forma:

```json
{ "success": true, "message": "Predicciones generadas para ...", "data": [ { } ] }
```

Errores: `400` si la entrada es inválida (dominio inexistente, datos insuficientes, etc.) y
`500` si el predictor no está disponible.

| Método | Ruta | Campos obligatorios |
|---|---|---|
| POST | `/predict/ipm` | `domain` |
| POST | `/predict/privacion` | `variable`, `domain` |
| POST | `/predict/porcentaje` | `domain` |
| POST | `/predict/dimension` | `dimension`, `domain` |
| POST | `/predict/incidence_by_sex` | `domain`, `sex` |
| POST | `/predict/incidence_by_head_sex` | `domain`, `sex` |
| POST | `/predict/department_poverty` | `region`, `department` |
| POST | `/predict/multiple_departments` | `region`, `departments[]` |
| POST | `/predict/household_poverty` | `region`, `department`, `personas_que_habitan_ese_hogar` y 15 campos `privacion_*` |
| POST | `/predict/multiple_households` | `households[]` (cada uno con los campos de `household_poverty`) |
| POST | `/predict/contribution_latam` | `privation`, `country` |
| POST | `/predict/national_poverty` | `country`, `area`, `measure` |
| POST | `/predict/poverty_by_age` | `country`, `area`, `age_range` |
| GET | `/` | Lista de endpoints |
| GET | `/health` | Estado de carga de cada predictor |

Ejemplo:

```bash
curl -X POST http://localhost:11010/predict/ipm \
  -H "Content-Type: application/json" \
  -d '{"domain": "Nacional", "begin_year": 2026, "years": 5}'
```

La respuesta trae en `data` un registro por año con las columnas `año`, `dominio` e `ipm_predicho`.

Notas de entrada:

- **Nivel 2:** región y departamento se comparan sin distinguir mayúsculas, tildes ni signos, y
  se aceptan alias (`region_name`, `departamento`, `departamentos`).
- `/predict/household_poverty` devuelve, además de la serie, `total_deprivations`,
  `initial_classification` (`Pobre` si el IPM inicial ≥ 0.3), `deprivation_level`
  (`Alto` ≥ 0.5, `Medio` ≥ 0.2, `Bajo`) y `trend` (`increasing`/`decreasing`).
- CORS está abierto a cualquier origen (`*`).

Payloads completos por nivel: archivos `Endpoints nivel 1`, `Endpoints nivel 2` y `Endpoints nivel 3`.
Dominios de nivel 1 disponibles: Nacional, Cabecera, Caribe, Central, Oriental, Pacífica,
Bogotá (cabecera), Agregado, Centros poblados y rural disperso. Dimensiones (vista 4):
Educación, Niñez y Juventud, Trabajo, Salud, Vivienda.

### Exportación a PDF

`src/api/pdf_export.py` expone `generate_pdf_report(title, data, subtitle=None, columns=None) -> bytes`,
que convierte una lista de registros (el campo `data` de una respuesta) en un PDF tabular apaisado.

## Docker y despliegue

```bash
docker build -t pato-ia-predictiva .
docker run -p 11010:11010 --env-file .env pato-ia-predictiva
```

La imagen parte de `python:3.12-slim`, instala `requirements.txt`, copia el repo (incluidos los modelos)
y ejecuta `python ./src/api/main.py`.

CI/CD en `.github/workflows/`:

1. **`main.yml` – Test and push container:** en cada push a `main` construye la imagen y la sube a
   Docker Hub con los tags `latest` y el SHA del commit (secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`).
2. **`deploy.yml` – Deploy container:** al terminar el anterior (o manualmente), un runner
   *self-hosted* ejecuta `docker compose pull && docker compose up -d` en `WORK_DIR`.

## Notas y limitaciones conocidas

- Los predictores consultan la base de datos en cada llamada a `predict`; si la BD cae, las predicciones fallan
  aunque los modelos estén cargados.
- La predicción es recursiva: el error se acumula a medida que crece `years`. Las métricas de arriba
  evalúan el conjunto de prueba, no horizontes largos.
- `pdf_export.py` existe pero **no está conectado** a ningún endpoint en `main.py`.
- `src/test_preprocessing.py` llama a `prepare_nivel1_data`, pero el método se llama `prepare_level1_data`
  (el script está desactualizado). El preprocesamiento de nivel 3 está pendiente en `preprocessing.py`.
- `Endpoints nivel 1` muestra `dimension` en `/predict/porcentaje`, pero el request real solo usa `domain`.
- `.gitignore` menciona `src/level2/` y `models/train_lv2_vw1/`, pero ambos están versionados.
- Las vistas 2 y 4 aparecen marcadas como "REVISAR" en `Endpoints nivel 1`.
