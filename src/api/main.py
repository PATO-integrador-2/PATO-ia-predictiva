# api/main.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from api.pdf_export import generate_pdf_report

# PREDICTORES 
# NIVEL 1

# 1. IPM por dominio
from level1.lv1_vw1_predict import DomainPredictor

# 2. Privaciones (15 variables)
from level1.lv1_vw2_predict import PrivationsPredictor

# 3. Porcentaje de privaciones
from level1.lv1_vw3_predict import AverageSimplePredictor

# 4. Contribuciones por dimensión
from level1.lv1_vw4_predict import DimensionContributionPredictor

# 5. Incidencia por sexo de la persona
from level1.lv1_vw5_predict import IncidenceBySexPredictor

# 6. Incidencia por sexo del jefe del hogar
from level1.lv1_vw6_predict import IncidenceByHeadSexPredictor

# NIVEL 2

#7 Pobreza por departamento
from level2.lv2_vw1_predict import DepartmentPovertyPredictor

# NIVEL 3
# 8. Contribución relativa de privaciones en Latinoamérica
from level3.lv3_vw1_predict import ContributionLatamPredictor

# 9. Población en pobreza multidimensional (cifras nacionales)
from level3.lv3_vw2_predict import NationalPovertyPredictor

# 10. Pobreza por grupo de edad y área
from level3.lv3_vw3_predict import PovertyByAgePredictorV2 #cambiar ese v2 feo
# MODELOS PYDANTIC (Request/Response)

# NIVEL 1 
class PredictIPMRequest(BaseModel):
    """Request para predicción de IPM."""
    domain: str
    begin_year: int = 2026
    years: int = 5


class PredictPrivacionRequest(BaseModel):
    """Request para predicción de privación."""
    variable: str
    domain: str
    begin_year: int = 2026
    years: int = 5


class PredictPorcentajeRequest(BaseModel):
    """Request para predicción de porcentaje."""
    domain: str
    begin_year: int = 2026
    years: int = 5


class PredictDimensionRequest(BaseModel):
    """Request para predicción de contribución por dimensión."""
    dimension: str
    domain: str
    begin_year: int = 2026
    years: int = 5


class PredictIncidenceBySexRequest(BaseModel):
    """Request para predicción de incidencia por sexo de la persona."""
    domain: str
    sex: str
    begin_year: int = 2026
    years: int = 5


class PredictIncidenceByHeadSexRequest(BaseModel):
    """Request para predicción de incidencia por sexo del jefe del hogar."""
    domain: str
    sex: str
    begin_year: int = 2026
    years: int = 5

#NIVEL 2 

class PredictDepartmentPovertyRequest(BaseModel):
    """Request para predicción de pobreza por departamento (múltiples años)."""
    region: str
    departamento: str
    begin_year: int = 2026
    years: int = 5


class PredictMultipleDepartmentsRequest(BaseModel):
    """Request para predicción de pobreza para múltiples departamentos (múltiples años)."""
    region: str
    departments: List[str]
    begin_year: int = 2026
    years: int = 5


class PredictHouseholdPovertyRequest(BaseModel):
    """Request para predicción de pobreza para un hogar específico."""
    region: str
    departamento: str
    personas_que_habitan_ese_hogar: int
    privacion_bajo_logro_educativo: int
    privacion_analfabetismo: int
    privacion_inasistencia_escolar: int
    privacion_rezago_escolar: int
    privacion_atencion_integral_primera_infancia: int
    privacion_trabajo_infantil: int
    privacion_no_aseguramiento_salud: int
    privacion_barreras_acceso_salud: int
    privacion_desempleo_larga_duracion: int
    privacion_tasa_empleo_formal: int
    privacion_no_acceso_agua_mejorada: int
    privacion_inadecuada_eliminacion_excretas: int
    privacion_inadecuado_material_pisos: int
    privacion_inadecuado_material_paredes: int
    privacion_hacinamiento_critico: int
    begin_year: int = 2026
    years: int = 5


class PredictMultipleHouseholdsRequest(BaseModel):
    """Request para predicción de pobreza para múltiples hogares (múltiples años)."""
    households: List[PredictHouseholdPovertyRequest]
    
#NIVEL 3
class PredictContributionLatamRequest(BaseModel):
    """Request para predicción de contribución de privaciones en Latinoamérica."""
    privation: str
    country: str
    begin_year: int = 2026
    years: int = 5


class PredictNationalPovertyRequest(BaseModel):
    """Request para predicción de pobreza multidimensional (cifras nacionales)."""
    country: str
    area: str
    measure: str
    begin_year: int = 2026
    years: int = 5


class PredictPovertyByAgeRequest(BaseModel):
    """Request para predicción de pobreza por grupo de edad y área."""
    country: str
    area: str
    age_range: str
    begin_year: int = 2026
    years: int = 5

# EXPORTACIÓN
class ExportPDFRequest(BaseModel):
    """Request para exportar un conjunto de datos (p. ej. el `data` de cualquier /predict/*) como PDF."""
    title: str
    subtitle: Optional[str] = None
    data: List[dict]
    columns: Optional[List[str]] = None


# Objects de respuesta de la API
class APIResponse(BaseModel):
    """Response general de la API."""
    success: bool
    message: str
    data: Optional[List] = None


# INICIALIZAR FASTAPI

app = FastAPI(
    title="IPM Predictor API",
    description="API para predicción de Pobreza Multidimensional",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CARGAR PREDICTORES

predictors = {}

#NIVEL 1 
print("CARGANDO PREDICTORES NIVEL 1")
# 1. IPM por dominio
try:
    predictors['ipm'] = DomainPredictor()
    print("   Predictor IPM cargado")
except Exception as e:
    print(f"   Error cargando predictor IPM: {e}")
    predictors['ipm'] = None

# 2. Privaciones
try:
    predictors['privacion'] = PrivationsPredictor(usar_global=True)
    print("   Predictor Privaciones cargado")
except Exception as e:
    print(f"   Error cargando predictor Privaciones: {e}")
    predictors['privacion'] = None

# 3. Porcentaje de privaciones
try:
    predictors['porcentaje'] = AverageSimplePredictor()
    print("   Predictor Porcentaje cargado")
except Exception as e:
    print(f"   Error cargando predictor Porcentaje: {e}")
    predictors['porcentaje'] = None

# 4. Contribuciones por dimensión
try:
    predictors['dimension'] = DimensionContributionPredictor()
    print("   Predictor Dimension cargado")
except Exception as e:
    print(f"   Error cargando predictor Dimension: {e}")
    predictors['dimension'] = None

# 5. Incidencia por sexo de la persona
try:
    predictors['incidence_sex'] = IncidenceBySexPredictor()
    print("   Predictor Incidencia por Sexo cargado")
except Exception as e:
    print(f"   Error cargando predictor Incidencia por Sexo: {e}")
    predictors['incidence_sex'] = None

# 6. Incidencia por sexo del jefe del hogar
try:
    predictors['incidence_head_sex'] = IncidenceByHeadSexPredictor()
    print("   Predictor Incidencia por Sexo del Jefe del Hogar cargado")
except Exception as e:
    print(f"   Error cargando predictor Incidencia por Sexo del Jefe del Hogar: {e}")
    predictors['incidence_head_sex'] = None

print("\n CARGANDO PREDICTORES NIVEL 3")

# NIVEL 2 
# 7 Pobreza por departamento
try:
    from level2.lv2_vw1_predict import DepartmentPovertyPredictor
    predictors['department_poverty'] = DepartmentPovertyPredictor()
    print("   Predictor Pobreza por Departamento cargado")
except Exception as e:
    print(f"   Error cargando predictor Pobreza por Departamento: {e}")
    predictors['department_poverty'] = None


# NIVEL 3 
# 7. Contribución relativa de privaciones en Latinoamérica
try:
    predictors['contribution_latam'] = ContributionLatamPredictor()
    print("   Predictor Contribución Latam cargado")
except Exception as e:
    print(f"   Error cargando predictor Contribución Latam: {e}")
    predictors['contribution_latam'] = None

# 8. Población en pobreza multidimensional (cifras nacionales)
try:
    predictors['national_poverty'] = NationalPovertyPredictor()
    print("   Predictor Pobreza Nacional cargado")
except Exception as e:
    print(f"   Error cargando predictor Pobreza Nacional: {e}")
    predictors['national_poverty'] = None

# 9. Pobreza por grupo de edad y área
try:
    predictors['poverty_by_age'] = PovertyByAgePredictorV2()
    print("   Predictor Pobreza por Edad cargado")
except Exception as e:
    print(f"   Error cargando predictor Pobreza por Edad: {e}")
    predictors['poverty_by_age'] = None
    
    
print("\nAPI lista para usar")


# ENDPOINTS POST

# NIVEL 1 

@app.post("/predict/ipm", response_model=APIResponse)
async def predict_ipm(request: PredictIPMRequest):
    """Predice IPM para un dominio."""
    if predictors['ipm'] is None:
        raise HTTPException(status_code=500, detail="Predictor IPM no disponible")

    try:
        df = predictors['ipm'].predict(domain=request.domain, begin_year=request.begin_year, years=request.years)
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.domain}",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/privacion", response_model=APIResponse)
async def predict_privacion(request: PredictPrivacionRequest):
    """Predice una privación específica."""
    if predictors['privacion'] is None:
        raise HTTPException(status_code=500, detail="Predictor Privaciones no disponible")

    try:
        df = predictors['privacion'].predict(
            variable=request.variable,
            domain=request.domain,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.variable} - {request.domain}",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/porcentaje", response_model=APIResponse)
async def predict_porcentaje(request: PredictPorcentajeRequest):
    """Predice porcentaje de privaciones."""
    if predictors['porcentaje'] is None:
        raise HTTPException(status_code=500, detail="Predictor Porcentaje no disponible")

    try:
        df = predictors['porcentaje'].predict(
            domain=request.domain,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.domain}",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/dimension", response_model=APIResponse)
async def predict_dimension(request: PredictDimensionRequest):
    """Predice contribución de una dimensión."""
    if predictors['dimension'] is None:
        raise HTTPException(status_code=500, detail="Predictor Dimension no disponible")

    try:
        df = predictors['dimension'].predict(
            dimension=request.dimension,
            domain=request.domain,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.dimension} - {request.domain}",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/incidence_by_sex", response_model=APIResponse)
async def predict_incidence_by_sex(request: PredictIncidenceBySexRequest):
    """Predice incidencia por sexo de la persona."""
    if predictors['incidence_sex'] is None:
        raise HTTPException(status_code=500, detail="Predictor Incidencia por Sexo no disponible")

    try:
        df = predictors['incidence_sex'].predict(
            domain=request.domain,
            sex=request.sex,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.sex} - {request.domain}",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/incidence_by_head_sex", response_model=APIResponse)
async def predict_incidence_by_head_sex(request: PredictIncidenceByHeadSexRequest):
    """Predice incidencia por sexo del jefe del hogar."""
    if predictors['incidence_head_sex'] is None:
        raise HTTPException(status_code=500, detail="Predictor Incidencia por Sexo del Jefe del Hogar no disponible")

    try:
        df = predictors['incidence_head_sex'].predict(
            domain=request.domain,
            sex=request.sex,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.sex} - {request.domain}",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# NIVEL 2
@app.post("/predict/department_poverty", response_model=APIResponse)
async def predict_department_poverty(request: PredictDepartmentPovertyRequest):
    """
    Predice el IPM promedio para un departamento específico para múltiples años.
    
    Args:
        region: Nombre de la región (ej. 'Caribe', 'Andina')
        departamento: Nombre del departamento (ej. 'Atlántico', 'Antioquia')
        begin_year: Año inicial para la predicción (default: 2026)
        years: Número de años a predecir (default: 5)
    
    Returns:
        Lista de IPMs predichos para el departamento
    """
    if predictors['department_poverty'] is None:
        raise HTTPException(status_code=500, detail="Predictor Pobreza por Departamento no disponible")

    try:
        # Obtener datos historicos para contexto
        historical = predictors['department_poverty']._get_historical_data(
            request.region, 
            request.departamento, 
            years_hist=5
        )
        
        # Predecir para cada año
        predictions = []
        for year in range(request.begin_year, request.begin_year + request.years):
            ipm_pred = predictors['department_poverty'].predict_department_average(
                region=request.region,
                departamento=request.departamento,
                year=year
            )
            predictions.append({
                'anio': year,
                'ipm_predicho': round(ipm_pred, 4)
            })
        
        # Calcular tendencia
        ultimo_real = historical['ipm'].iloc[-1]
        ultimo_predicho = predictions[-1]['ipm_predicho']
        
        data = [{
            'region': request.region,
            'departamento': request.departamento,
            'begin_year': request.begin_year,
            'years': request.years,
            'ipm_historico': [
                {
                    'anio': int(row['anio']),
                    'ipm': float(row['ipm'])
                }
                for _, row in historical.iterrows()
            ],
            'predicciones': predictions,
            'tendencia': 'decreciente' if ultimo_predicho < ultimo_real else 'creciente',
            'cambio_porcentual': round(((ultimo_predicho - ultimo_real) / ultimo_real) * 100, 2) if ultimo_real > 0 else 0
        }]

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.departamento} - {request.region} ({request.years} años)",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/multiple_departments", response_model=APIResponse)
async def predict_multiple_departments(request: PredictMultipleDepartmentsRequest):
    """
    Predice el IPM para múltiples departamentos de una región para múltiples años.
    
    Args:
        region: Nombre de la región
        departments: Lista de nombres de departamentos
        begin_year: Año inicial para la predicción (default: 2026)
        years: Número de años a predecir (default: 5)
    
    Returns:
        DataFrame comparativo con predicciones para todos los departamentos
    """
    if predictors['department_poverty'] is None:
        raise HTTPException(status_code=500, detail="Predictor Pobreza por Departamento no disponible")

    try:
        # Predecir para cada departamento
        resultados_por_depto = {}
        
        for depto in request.departments:
            try:
                # Obtener datos historicos
                historical = predictors['department_poverty']._get_historical_data(
                    request.region, 
                    depto, 
                    years_hist=3
                )
                
                # Predecir para cada año
                predicciones = []
                for year in range(request.begin_year, request.begin_year + request.years):
                    ipm_pred = predictors['department_poverty'].predict_department_average(
                        region=request.region,
                        departamento=depto,
                        year=year
                    )
                    predicciones.append({
                        'anio': year,
                        'ipm_predicho': round(ipm_pred, 4)
                    })
                
                resultados_por_depto[depto] = {
                    'ipm_historico': float(historical['ipm'].iloc[-1]),
                    'anio_historico': int(historical['anio'].iloc[-1]),
                    'predicciones': predicciones
                }
                
            except Exception as e:
                print(f"  Error con {depto}: {e}")
                resultados_por_depto[depto] = {
                    'error': str(e)
                }
        
        # Crear tabla comparativa por años
        tabla_comparativa = []
        for year in range(request.begin_year, request.begin_year + request.years):
            fila = {'anio': year}
            for depto in request.departments:
                if depto in resultados_por_depto and 'predicciones' in resultados_por_depto[depto]:
                    # Buscar el valor para este año
                    for pred in resultados_por_depto[depto]['predicciones']:
                        if pred['anio'] == year:
                            fila[depto] = pred['ipm_predicho']
                            break
                else:
                    fila[depto] = None
            tabla_comparativa.append(fila)
        
        data = [{
            'region': request.region,
            'begin_year': request.begin_year,
            'years': request.years,
            'departamentos': request.departments,
            'resultados_por_departamento': resultados_por_depto,
            'tabla_comparativa': tabla_comparativa
        }]

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {len(request.departments)} departamentos",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/household_poverty", response_model=APIResponse)
async def predict_household_poverty(request: PredictHouseholdPovertyRequest):
    """
    Predice el IPM para un hogar específico basado en sus características.
    
    Args:
        Todos los campos de privación del hogar (16 variables)
        begin_year: Año inicial para la predicción (default: 2026)
        years: Número de años a predecir (default: 5)
    
    Returns:
        IPM predicho para el hogar en múltiples años
    """
    if predictors['department_poverty'] is None:
        raise HTTPException(status_code=500, detail="Predictor Pobreza por Departamento no disponible")

    try:
        # Convertir request a diccionario
        hogar = request.dict()
        
        # Predecir para cada año
        predicciones = []
        for year in range(request.begin_year, request.begin_year + request.years):
            # Actualizar el año en el hogar (aunque no se usa directamente, lo mantenemos para contexto)
            hogar['anio'] = year
            
            # Predecir
            ipm_pred = predictors['department_poverty'].predict_household(hogar)
            
            predicciones.append({
                'anio': year,
                'ipm_predicho': round(ipm_pred, 4)
            })
        
        # Calcular número de privaciones
        privaciones = [
            hogar['privacion_bajo_logro_educativo'],
            hogar['privacion_analfabetismo'],
            hogar['privacion_inasistencia_escolar'],
            hogar['privacion_rezago_escolar'],
            hogar['privacion_atencion_integral_primera_infancia'],
            hogar['privacion_trabajo_infantil'],
            hogar['privacion_no_aseguramiento_salud'],
            hogar['privacion_barreras_acceso_salud'],
            hogar['privacion_desempleo_larga_duracion'],
            hogar['privacion_tasa_empleo_formal'],
            hogar['privacion_no_acceso_agua_mejorada'],
            hogar['privacion_inadecuada_eliminacion_excretas'],
            hogar['privacion_inadecuado_material_pisos'],
            hogar['privacion_inadecuado_material_paredes'],
            hogar['privacion_hacinamiento_critico']
        ]
        total_privaciones = sum(privaciones)
        
        # Clasificación basada en el primer año de predicción
        ipm_inicial = predicciones[0]['ipm_predicho'] if predicciones else 0
        
        data = [{
            'region': hogar['region'],
            'departamento': hogar['departamento'],
            'personas_en_hogar': hogar['personas_que_habitan_ese_hogar'],
            'total_privaciones': total_privaciones,
            'begin_year': request.begin_year,
            'years': request.years,
            'predicciones': predicciones,
            'clasificacion_inicial': 'Pobre' if ipm_inicial >= 0.3 else 'No Pobre',
            'nivel_privacion': 'Alto' if ipm_inicial >= 0.5 else 'Medio' if ipm_inicial >= 0.2 else 'Bajo',
            'tendencia': 'decreciente' if len(predicciones) >= 2 and predicciones[-1]['ipm_predicho'] < predicciones[0]['ipm_predicho'] else 'creciente'
        }]

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para el hogar en {hogar['departamento']} ({request.years} años)",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/multiple_households", response_model=APIResponse)
async def predict_multiple_households(request: PredictMultipleHouseholdsRequest):
    """
    Predice el IPM para múltiples hogares para múltiples años.
    
    Args:
        households: Lista de hogares con sus características
    
    Returns:
        Lista de predicciones para cada hogar en múltiples años
    """
    if predictors['department_poverty'] is None:
        raise HTTPException(status_code=500, detail="Predictor Pobreza por Departamento no disponible")

    try:
        resultados = []
        
        for idx, hogar_request in enumerate(request.households):
            # Convertir a diccionario
            hogar = hogar_request.dict()
            
            # Predecir para cada año
            predicciones = []
            for year in range(hogar_request.begin_year, hogar_request.begin_year + hogar_request.years):
                hogar['anio'] = year
                ipm_pred = predictors['department_poverty'].predict_household(hogar)
                predicciones.append({
                    'anio': year,
                    'ipm_predicho': round(ipm_pred, 4)
                })
            
            # Calcular privaciones
            privaciones = [
                hogar['privacion_bajo_logro_educativo'],
                hogar['privacion_analfabetismo'],
                hogar['privacion_inasistencia_escolar'],
                hogar['privacion_rezago_escolar'],
                hogar['privacion_atencion_integral_primera_infancia'],
                hogar['privacion_trabajo_infantil'],
                hogar['privacion_no_aseguramiento_salud'],
                hogar['privacion_barreras_acceso_salud'],
                hogar['privacion_desempleo_larga_duracion'],
                hogar['privacion_tasa_empleo_formal'],
                hogar['privacion_no_acceso_agua_mejorada'],
                hogar['privacion_inadecuada_eliminacion_excretas'],
                hogar['privacion_inadecuado_material_pisos'],
                hogar['privacion_inadecuado_material_paredes'],
                hogar['privacion_hacinamiento_critico']
            ]
            
            ipm_inicial = predicciones[0]['ipm_predicho'] if predicciones else 0
            
            resultados.append({
                'hogar_id': idx + 1,
                'region': hogar['region'],
                'departamento': hogar['departamento'],
                'personas_en_hogar': hogar['personas_que_habitan_ese_hogar'],
                'total_privaciones': sum(privaciones),
                'begin_year': hogar_request.begin_year,
                'years': hogar_request.years,
                'predicciones': predicciones,
                'clasificacion_inicial': 'Pobre' if ipm_inicial >= 0.3 else 'No Pobre'
            })

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {len(request.households)} hogares",
            data=resultados
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# NIVEL 3 

@app.post("/predict/contribution_latam", response_model=APIResponse)
async def predict_contribution_latam(request: PredictContributionLatamRequest):
    """Predice contribución relativa de privaciones en Latinoamérica."""
    if predictors['contribution_latam'] is None:
        raise HTTPException(status_code=500, detail="Predictor Contribución Latam no disponible")

    try:
        df = predictors['contribution_latam'].predict(
            privation=request.privation,
            country=request.country,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.privation} - {request.country}",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/national_poverty", response_model=APIResponse)
async def predict_national_poverty(request: PredictNationalPovertyRequest):
    """Predice población en situación de pobreza multidimensional (cifras nacionales)."""
    if predictors['national_poverty'] is None:
        raise HTTPException(status_code=500, detail="Predictor Pobreza Nacional no disponible")

    try:
        df = predictors['national_poverty'].predict(
            country=request.country,
            area=request.area,
            measure=request.measure,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.measure} - {request.country} ({request.area})",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/poverty_by_age", response_model=APIResponse)
async def predict_poverty_by_age(request: PredictPovertyByAgeRequest):
    """Predice incidencia de pobreza multidimensional por grupo de edad y área."""
    if predictors['poverty_by_age'] is None:
        raise HTTPException(status_code=500, detail="Predictor Pobreza por Edad no disponible")

    try:
        df = predictors['poverty_by_age'].predict(
            country=request.country,
            area=request.area,
            age_range=request.age_range,
            begin_year=request.begin_year,
            years=request.years
        )
        data = df.to_dict(orient='records')

        return APIResponse(
            success=True,
            message=f"Predicciones generadas para {request.age_range} - {request.country} ({request.area})",
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# EXPORTACIÓN

@app.post("/export/pdf")
async def export_pdf(request: ExportPDFRequest):
    """
    Exporta a PDF un conjunto de datos generado por cualquier endpoint /predict/*.

    El cliente primero consulta el endpoint de predicción deseado y luego
    envía el campo `data` de esa respuesta (junto con un título) a este
    endpoint para obtener un archivo PDF descargable.
    """
    try:
        pdf_bytes = generate_pdf_report(
            title=request.title,
            data=request.data,
            subtitle=request.subtitle,
            columns=request.columns
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = f"{request.title.strip().replace(' ', '_') or 'reporte'}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ENDPOINTS GET

@app.get("/")
async def root():
    return {
        "name": "IPM Predictor API",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            # NIVEL 1
            "POST /predict/ipm": "Predicción de Incidencia de IPM",
            "POST /predict/privacion": "Predicción de privación por hogar según variable",
            "POST /predict/porcentaje": "Predicción de porcentaje de privaciones entre las personas en situación de Pobreza Multidimensional",
            "POST /predict/dimension": "Predicción de contribución a la incidencia Ajustada del IPM por dimensión",
            "POST /predict/incidence_by_sex": "Predicción de incidencia por sexo",
            "POST /predict/incidence_by_head_sex": "Predicción de incidencia por sexo del jefe del hogar",
            # NIVEL 2
            "POST /predict/department_poverty": "Predicción de IPM por departamento",
            "POST /predict/multiple_departments": "Predicción de IPM para múltiples departamentos",
            "POST /predict/household_poverty": "Predicción de IPM para un hogar específico",
            "POST /predict/multiple_households": "Predicción de IPM para múltiples hogares",
            # NIVEL 3
            "POST /predict/contribution_latam": "Predicción de contribución relativa de privaciones en Latinoamérica",
            "POST /predict/national_poverty": "Predicción de población en situación de pobreza multidimensional (cifras nacionales)",
            "POST /predict/poverty_by_age": "Predicción de incidencia de pobreza multidimensional por grupo de edad y área",
            # EXPORTACIÓN
            "POST /export/pdf": "Exporta a PDF los datos obtenidos de cualquier endpoint /predict/*"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "predictors": {
            "ipm": predictors.get('ipm') is not None,
            "privacion": predictors.get('privacion') is not None,
            "porcentaje": predictors.get('porcentaje') is not None,
            "dimension": predictors.get('dimension') is not None,
            "incidence_sex": predictors.get('incidence_sex') is not None,
            "incidence_head_sex": predictors.get('incidence_head_sex') is not None,
            "contribution_latam": predictors.get('contribution_latam') is not None,
            "national_poverty": predictors.get('national_poverty') is not None,
            "poverty_by_age": predictors.get('poverty_by_age') is not None,
            "department_poverty": predictors.get('department_poverty') is not None
        }
    } 


# EJECUCIÓN

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=os.getenv("PORT"),
        reload=True
    )