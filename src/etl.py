import pandas as pd
import sqlite3

# Rutas de los archivos CSV
ruta_clasificacion = r"./data/raw/dataset_clasificacion.csv"
ruta_regresion = r"./data/raw/dataset_regresion.csv"

# Rutas de las bases de datos
db_clasificacion = r"./database/clasificacion.db"
db_regresion = r"./database/regresion.db"

# Leer CSV de clasificación
df_clasificacion = pd.read_csv(ruta_clasificacion, sep=";")

# Leer CSV de regresión
df_regresion = pd.read_csv(ruta_regresion, sep=";")

# Conexión a SQLite para clasificación
conn_clasificacion = sqlite3.connect(db_clasificacion)

# Importar a tabla SQLite
df_clasificacion.to_sql("clasificacion", conn_clasificacion, if_exists="replace", index=False)

# Cerrar conexión
conn_clasificacion.close()

# Conexión a SQLite para regresión
conn_regresion = sqlite3.connect(db_regresion)

# Importar a tabla SQLite
df_regresion.to_sql("regresion", conn_regresion, if_exists="replace", index=False)

# Cerrar conexión
conn_regresion.close()

print("Importación completada con éxito.")
print("Tabla 'clasificacion' cargada en database/clasificacion.db")
print("Tabla 'regresion' cargada en database/regresion.db")