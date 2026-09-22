# Conexión a la base de datos

import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()  # Cargar variables de entorno desde el archivo .env

class Database:
    def __init__(self): 
        # Cargar las variables de entorno para la conexión a la base de datos
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.database = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")

        #Conección a la base de datos
        self.connection_string = (
            f"postgresql://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.database}"
        )

        self.engine = create_engine(
            self.connection_string,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )

        print("Conexión a la base de datos establecida.")

        #prueba de conexión 
    def test_connection(self) -> bool:
        try: 
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                print("Prueba de conexión exitosa.")
                return True
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            return False

        # Método para ejecutar consultas SQL y devolver un DataFrame

    def execute_query(self, query:str) -> pd.DataFrame:
        with self.engine.connect() as connection:
            return pd.read_sql(text(query), connection)

    # NIVEL 1 

    # vista 1 public.vw_ipm_by_domain
    def get_data_view_ipm_domain(self) -> pd.DataFrame:
        #query = "SELECT * FROM public.vw_ipm_by_domain"
        #return self.execute_query(query)
        query = "SELECT * FROM public.vw_ipm_by_domain"
        data_frame = self.execute_query(query)
        data_frame = data_frame[~data_frame['dominio'].str.contains('__test__', na=False)] # Filtrar filas que contienen '_test_')
        return data_frame
    #vista 2 public.vw_deprivations_by_variable
    def get_data_view_deprivations_variable(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_deprivations_by_variable"
        return self.execute_query(query)
    #vista 3 public.vw_average_deprivations
    def get_data_view_average_deprivations(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_average_deprivations"
        return self.execute_query(query)
    #vista 4 public.vw_dimension_contribution
    def get_data_view_dimension_contribution(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_dimension_contribution"
        return self.execute_query(query)
    #vista 5 public.vw_incidence_by_person_sex
    def get_data_view_incidence_by_person_sex(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_incidence_by_person_sex"
        return self.execute_query(query)
    #vista 6 public.vw_incidence_by_household_head_sex
    def get_data_view_incidence_by_household_head_sex(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_incidence_by_household_head_sex"
        return self.execute_query(query)
    
    #NIVEL 2  
    #vista 7 public.vw_household_poverty_by_department
    def get_data_view_household_poverty_by_department(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_household_poverty_by_department"
        return self.execute_query(query)
    
    #NIVEL 3 
    
    #vista 8 vw.dashboard03_deprivation_contribution
    def get_data_view_dashboard03_deprivation_contribution(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_dashboard03_deprivation_contribution"
        return self.execute_query(query)
    #vista 9 vw_dashboard03_national_poverty
    def get_data_view_dashboard03_national_poverty(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_dashboard03_national_poverty"
        return self.execute_query(query)
    #vista 10 vw_dashboard03_poverty_by_age
    def get_data_view_dashboard03_poverty_by_age(self) -> pd.DataFrame:
        query = "SELECT * FROM public.vw_dashboard03_poverty_by_age"
        return self.execute_query(query)

    # Prueba de la clase Database
    def test_database(self):
        return self.test_connection()

if __name__ == "__main__":
    database = Database()
    if database.test_connection():
        df_ipm_domain = database.get_data_view_ipm_domain()
        print('Vista 1 - Cargado correctamente')
        df_deprivations_variable = database.get_data_view_deprivations_variable()
        print('Vista 2 - Cargado correctamente')
        df_average_deprivations = database.get_data_view_average_deprivations()
        print('Vista 3 - Cargado correctamente')
        df_dimension_contribution = database.get_data_view_dimension_contribution()
        print('Vista 4 - Cargado correctamente')
        df_incidence_by_person_sex = database.get_data_view_incidence_by_person_sex()
        print('Vista 5 - Cargado correctamente')
        df_incidence_by_household_head_sex = database.get_data_view_incidence_by_household_head_sex()
        print('Vista 6 - Cargado correctamente')
        df_household_poverty_by_department = database.get_data_view_household_poverty_by_department()
        print('Vista 7 - Cargado correctamente')
        df_deprivation_contribution = database.get_data_view_dashboard03_deprivation_contribution()
        print('Vista 8 - Cargado correctamente')
        df_national_poverty = database.get_data_view_dashboard03_national_poverty()
        print('Vista 9 - Cargado correctamente')
        df_poverty_by_age = database.get_data_view_dashboard03_poverty_by_age()
        print('Vista 10 - Cargado correctamente')
        print("Conectado y vistas obtenidas correctamente")
    else:
        print("Error al conectar a la base de datos.")