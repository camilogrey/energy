# Importacion de sqlalchemy
import os
paquete = "sqlalchemy"

try:
    exec("import " +  paquete)                    # ejecuta la importación del paquete para verificar si ya está instalado
    print(paquete + " ya está instalado")
except:                                           # si obtiene un error ...
    print("Instalando paquete " + paquete)
    os.system("python -m pip install " + paquete) # ejecuta su instalación
    
# Importacion de Librerias (Verificación la Versión)

import sqlalchemy
import os
import pandasql as sqldf

version_sqlalchemy = sqlalchemy.__version__  ## Almacena como texto la versión actual de sqlalchemy
print("la versión instalada de sqlalchemy es la: ", version_sqlalchemy)

version_mayor = int(version_sqlalchemy.split(".")[0])  ## Extrae el primer número de la versión (índice 0)
print(version_mayor)

paquete = "pansql" ## por defecto trata de instalar pansql

if version_mayor >= 2:                          # Prueba si la versión mayor de sqlalchemy es igual o superior a 2
  os.system("python -m pip install " + paquete) # de ser así instala el paquete  "pansql"
  print("Se instaló " + paquete)                # Confirma la instalación del paquete
else:                                           # De lo contario (la versión mayor es inferior a 2)
  paquete = "pandasql"                          # cambia el paquete objetivo a "pandasql"
  os.system("python -m pip install " + paquete) # instala el paquete  "pandasql"
  print("Se instaló " + paquete)                # Confirma la instalación del paquete

exec('from ' + paquete +  ' import sqldf')      # Carga la función que permite ejecutar SQL con dataframes de pandas
print("Se cargó la función sqldf")


# Carga de paquetes
import pandas as pd   # transformación y manipúlación de datos
import numpy  as np   # cálculos numéricos

# Requerido para la funcionalidad básica de gráficos de Python
import matplotlib.pyplot as plt


# Requerido para formatear fechas más adelante en el caso
import datetime
import matplotlib.dates as mdates

# Requerido para mostrar la imagen en una sola línea
from IPython.display import Image

# Funcionalidad de trazado avanzada con seaborn
import seaborn as sns

# puede establecer el estilo gráfico seaborn
sns.set(style="whitegrid")

import plotly.express as pex  ## Gráficos avanzados interactivos
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width', 20)
pd.set_option('max_colwidth', None)



## crea el dataframe para el consumo

energy_df = pd.read_csv("C:/Users/camil/Documents/Data Sciences/Own Projects/Energy/energy_consumption.csv"  ## Ruta relativa del archivo
                        , sep = ","                    ## Carater seprador de columnas
                        , header=0                     ## Línea donde están los encabezados de las columnas
                        )

## creando el dataframe para la generación
electricity_df = pd.read_csv("C:/Users/camil/Documents/Data Sciences/Own Projects/Energy/electricity_generation.csv"  ## Ruta relativa del archivo
                        , sep = ","                             ## Carater seprador de columnas
                        , header=0                              ## Línea donde están los encabezados de las columnas
                        )


print(energy_df.describe(include="all"))

print(electricity_df.describe(include="all"))

"""
# Desplegar los datos de energía (consumo)
print(energy_df.head(3))
"""

print(energy_df.head())
## Mismo resultado usando SQLite: SQLalchemy y pansql
n_datos = 3
result=sqldf(f"""SELECT *
                    FROM energy_df
                    LIMIT {n_datos}""")

print(result)

# Obtenga todas las descripciones únicas disponibles
print(len(energy_df["Description"].unique()))

energy_df.Description.unique()

## Mismo resultado usando SQLite:
result=sqldf(f"""SELECT DISTINCT Description
         FROM energy_df""")#.values
         
print(result)

# Desplegar los datos de electricidad (consumo)
print(electricity_df.head())


n_datos = 5
result=sqldf(f"""SELECT *
                    FROM electricity_df
                    LIMIT {n_datos}""")
                    
print(result)    


# Get all unique descriptions available
print(electricity_df["Description"].unique()) 


result=sqldf(f"""SELECT DISTINCT Description
         FROM electricity_df""")#.values

print(result)            
print("____________")                   

               
n_datos = 15
result=sqldf(f"""SELECT *
                    FROM electricity_df
                    LIMIT {n_datos}""")
                    
print(result)    

# Get all unique descriptions available
print(electricity_df["Description"].unique())

print(energy_df.info())
print(" --------------------------------------------------------")
print(electricity_df.info())


import sys
## Cualquiera de las dos formas es posible:
print(sys.getsizeof(energy_df)/1024) ## Peso en KB
energy_df.__sizeof__()/1024          ## Peso en KB

print(n_datos.__sizeof__())


energy_df["Description"]  = energy_df["Description"].astype("category")
energy_df["MSN"]          = energy_df["MSN"].astype("category")
energy_df["Unit"]         = energy_df["Unit"].astype("category")
energy_df["YYYYMM"]       = energy_df["YYYYMM"].astype(np.uint32)
energy_df["Column_Order"] = energy_df["Column_Order"].astype(np.uint8)
energy_df["Value"]        = energy_df["Value"].astype(np.float32)

print(energy_df.info())

print(" --------------------------------------------------------")
print(energy_df["Value"].min())
print(energy_df["Value"].max())

print("Reducción de memoria RAM: ", round(100*(82.5  - 323.6 )/323.6,1), "%")
# energy_df.describe()

print(np.finfo(np.float32).max)

print(np.iinfo(np.uint64))

print(" electricty")

import sys
## Cualquiera de las dos formas es posible:
print(sys.getsizeof(electricity_df)/1024) ## Peso en KB
electricity_df.__sizeof__()/1024          ## Peso en KB

print(n_datos.__sizeof__())

print(electricity_df.info())
print(electricity_df["Value"].head())


electricity_df["Description"]  = electricity_df["Description"].astype("category")
electricity_df["MSN"]          = electricity_df["MSN"].astype("category")
electricity_df["Unit"]         = electricity_df["Unit"].astype("category")
electricity_df["YYYYMM"]       = electricity_df["YYYYMM"].astype(np.uint32)
electricity_df["Column_Order"] = electricity_df["Column_Order"].astype(np.uint8)

"""
verificar el cambio de tipo de la columna
energy_df["Value"]        = energy_df["Value"].astype(np.float32)
electricity_df['Value'] = pd.to_numeric(electricity_df['Value'], errors='coerce')
"""

print(electricity_df.info())
"""
print(" --------------------------------------------------------")
print(electricity_df["Value"].min())
print(electricity_df["Value"].max())
"""
print("Reducción de memoria RAM: ", round(100*(385.3 - 129.9 )/385.3,1), "%")
# electricity_df.describe()


print(np.finfo(np.float32).max)

print(np.iinfo(np.uint64))


print(electricity_df[electricity_df["Value"] == "Not Available"]) ## sintaxis de PANDAS


         
print(result)
"""

## Otra forma
print(electricity_df.query("Value == Not Available"))

print(electricity_df[electricity_df["Value"] == "Not Available"].groupby(["Description"]).count())
"""

## Definir una nueva función personalizada
def string_to_float(r):
    return np.float32(r["Value"]) if r["Value"] != "Not Available" else None


# energy_df["Value"]      = energy_df.apply(string_to_float, axis=1)
electricity_df["Value"] = electricity_df.apply(string_to_float, axis=1)
electricity_df[1896:]

print(electricity_df[1896:])


def string_to_float_column(x):
    return np.float32(x) if x != "Not Available" else None


# Note that we don't need ```axis=1``` in the following example because we are
# operating on the column directly which is a Series:

# energy_df["Value"]      = energy_df.apply(string_to_float, axis=1)
# energy_df["Value"]        = energy_df["Value"].apply(string_to_float_column)
electricity_df["Value"]   = electricity_df["Value"].apply(string_to_float_column)

print(electricity_df.info())

result =sqldf("""
      SELECT * FROM electricity_df
      WHERE Value IS NULL
      """)
      
print(result.head())      

print(electricity_df[  electricity_df["Value"].isnull()])
print("XXXXXXXXXXXXXXXXX-")
print(electricity_df["Value"].apply(lambda x: np.float32(x) if x != "Not Available" else None))

## Misma transformación usando la función CAST  de SQLite:
sqldf("""SELECT MSN, YYYYMM, CASE
                                 WHEN Value <> 'Not Available' THEN CAST(Value AS REAL)
                                 --WHEN ............   THEN
                                 --WHEN ...........    THEN
                                 ELSE NULL
                             END AS Value
                 ,Column_Order
                 , Description
                 , Unit
         FROM electricity_df
     """).query('Value.isnull()') ## El método query( ) de pandas también sirve para consultas sencillas
     

# print(energy_df.shape)
830/electricity_df.shape[0]

# energy_df = energy_df.dropna()
electricity_df = electricity_df.dropna()

# print(energy_df.shape)
print(electricity_df.shape)
print(830/8216)

print(electricity_df[  electricity_df["Value"].isnull()])

result = sqldf("""
      SELECT * FROM electricity_df
      WHERE Value IS NULL
      """)
print(result)

print(int("195012"[:4]))
int("195012"[4:])

print(np.iinfo(np.uint16))

electricity_df["YYYY"] = electricity_df["YYYYMM"].apply(lambda x: np.uint16(str(x)[:4]) )
electricity_df["MM"]   = electricity_df["YYYYMM"].apply(lambda x: np.uint8(str(x)[4:]) )
print(electricity_df.info())

result = sqldf("""SELECT *
                       , CAST(SUBSTR(YYYYMM,5,6) AS INT) AS  MM
                       , CAST(SUBSTR(YYYYMM,1,4) AS INT) AS YYYY
         FROM electricity_df""")
print(result)

print("ENERGY_______________________________")

print(energy_df[energy_df["Value"] == "Not Available"]) ## sintaxis de PANDAS


         
print(result)
"""

## Otra forma
print(energy_df.query("Value == Not Available"))

print(energy_df[energy_df["Value"] == "Not Available"].groupby(["Description"]).count())
"""

## Definir una nueva función personalizada
def string_to_float(r):
    return np.float32(r["Value"]) if r["Value"] != "Not Available" else None


# energy_df["Value"]      = energy_df.apply(string_to_float, axis=1)
energy_df["Value"] = energy_df.apply(string_to_float, axis=1)
energy_df[1896:]

print(energy_df[1896:])


def string_to_float_column(x):
    return np.float32(x) if x != "Not Available" else None


# Note that we don't need ```axis=1``` in the following example because we are
# operating on the column directly which is a Series:

# energy_df["Value"]      = energy_df.apply(string_to_float, axis=1)
# energy_df["Value"]        = energy_df["Value"].apply(string_to_float_column)
energy_df["Value"]   = energy_df["Value"].apply(string_to_float_column)

print(energy_df.info())

result =sqldf("""
      SELECT * FROM energy_df
      WHERE Value IS NULL
      """)
      
print(result.head())      

print(energy_df[  energy_df["Value"].isnull()])
print("XXXXXXXXXXXXXXXXX-")
print(energy_df["Value"].apply(lambda x: np.float32(x) if x != "Not Available" else None))

## Misma transformación usando la función CAST  de SQLite:
sqldf("""SELECT MSN, YYYYMM, CASE
                                 WHEN Value <> 'Not Available' THEN CAST(Value AS REAL)
                                 --WHEN ............   THEN
                                 --WHEN ...........    THEN
                                 ELSE NULL
                             END AS Value
                 ,Column_Order
                 , Description
                 , Unit
         FROM energy_df
     """).query('Value.isnull()') ## El método query( ) de pandas también sirve para consultas sencillas
     

# print(energy_df.shape)
830/energy_df.shape[0]

# energy_df = energy_df.dropna()
energy_df= energy_df.dropna()

# print(energy_df.shape)
print(energy_df.shape)
print(830/8216)

print(energy_df[  energy_df["Value"].isnull()])

result = sqldf("""
      SELECT * FROM energy_df
      WHERE Value IS NULL
      """)
print(result)

print(int("195012"[:4]))
int("195012"[4:])

print(np.iinfo(np.uint16))

energy_df["YYYY"] = energy_df["YYYYMM"].apply(lambda x: np.uint16(str(x)[:4]) )
energy_df["MM"]   = energy_df["YYYYMM"].apply(lambda x: np.uint8(str(x)[4:]) )
print(energy_df.info())

result = sqldf("""SELECT *
                       , CAST(SUBSTR(YYYYMM,5,6) AS INT) AS  MM
                       , CAST(SUBSTR(YYYYMM,1,4) AS INT) AS YYYY
         FROM energy_df""")
print(result)

# Renombrar descripciones para los datos de energía
energy_short_dict = {
    "Primary Energy Consumed by the Commercial Sector": "PEC Commercial Sector",
    "Primary Energy Consumed by the Electric Power Sector": "PEC Electric Power Sector",
    "Primary Energy Consumed by the Industrial Sector": "PEC Industrial Sector",
    "Primary Energy Consumed by the Residential Sector": "PEC Residential Sector",
    "Primary Energy Consumed by the Transportation Sector": "PEC Transportation Sector",
    "Primary Energy Consumption Total": "PEC Total",
    "Total Energy Consumed by the Commercial Sector": "TEC Commercial Sector",
    "Total Energy Consumed by the Industrial Sector": "TEC Industrial Sector",
    "Total Energy Consumed by the Residential Sector": "TEC Residential Sector",
    "Total Energy Consumed by the Transportation Sector": "TEC Transportation Sector",
}


                          'Energy Consumption Balancing Item',
                          'Primary Energy Consumed by the Commercial Sector',
                          'Primary Energy Consumed by the Electric Power Sector',
                          'Primary Energy Consumed by the Industrial Sector',
                          ...,
                          'Total Energy Consumed by the Commercial Sector',
                          'Total Energy Consumed by the Industrial Sector',
                          'Total Energy Consumed by the Residential Sector',
                          'Total Energy Consumed by the Transportation Sector'

print(energy_df.columns)
print(energy_df["Description"].unique)

"""
# eliminar descripción no deseada
clean_energy_df = energy_df[
    energy_df["Description"].isin(energy_short_dict.keys())
].copy()

# remover columnas
remove_columns = ["MSN", "Column_Order"]

clean_energy_df = clean_energy_df.drop(remove_columns, axis=1) ## axis = 1 (columa)


# Limpiar nombres acortando la descripción
clean_energy_df["Description"] = clean_energy_df["Description"].apply(
    lambda x: energy_short_dict[x])

print(clean_energy_df.head())
"""