# Documentación base de datos de clasificación

## A) Nombre de la base de datos
**Titanic Dataset – Titanic Base de Datos**

## B) Fuente (URL)
https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

## C) Descripción general del problema
La base de datos Titanic contiene información sobre los pasajeros que viajaban en el **RMS Titanic**, un barco británico que se hundió en el océano Atlántico en abril de 1912. El hundimiento del Titanic es uno de los naufragios más famosos de la historia.

El **15 de abril de 1912**, durante su viaje inaugural, el RMS Titanic, considerado ampliamente como “insumergible”, se hundió después de chocar con un iceberg. Lamentablemente, no había suficientes botes salvavidas para todas las personas a bordo, lo que provocó la muerte de **1502 de los 2224 pasajeros y tripulantes**.

Aunque hubo cierto elemento de suerte en la supervivencia, parece que algunos grupos de personas tenían mayor probabilidad de sobrevivir que otros.

### ¿De dónde se recolectaron los datos?
Los datos provienen de:

- Registros históricos del viaje del Titanic
- Listas oficiales de pasajeros
- Informes de supervivencia posteriores al accidente

Posteriormente, estos datos fueron organizados y digitalizados por investigadores y comunidades académicas de análisis de datos, especialmente **UCI** y **Kaggle**.

Hoy en día se usan como:

- Base educativa
- Competencia de *machine learning*
- Ejemplo clásico de clasificación

### ¿Quién realizó el estudio?
No fue un estudio experimental moderno. Los datos fueron:

- Recopilados originalmente por autoridades marítimas británicas
- Documentados en informes oficiales del desastre
- Estructurados posteriormente por científicos de datos y universidades

Actualmente, la base es difundida por:

- Kaggle
- UCI Machine Learning Repository

## D) Objetivo del análisis
En este desafío, se solicita construir un modelo predictivo que responda a la pregunta:

> **¿Qué tipo de personas tenían más probabilidades de sobrevivir?**

Para ello, se utilizan datos de los pasajeros, por ejemplo:

- nombre
- edad
- género
- clase socioeconómica
- entre otros

Cada fila representa un pasajero y contiene variables demográficas, económicas y de ubicación dentro del barco.

## E) Variable objetivo (variable respuesta)
La variable objetivo del estudio es **`Survived`**.

Es una variable categórica binaria:

- `0` → No sobrevivió
- `1` → Sobrevivió

Por tanto, se trata de un **problema de clasificación supervisada**.

## F) Diccionario de variables

### `PassengerId`
- **Tipo:** Variable cualitativa nominal (identificador)
- **Describe:** Número único asignado a cada pasajero
- **Significado:** Identificación del pasajero

### `Survived`
- **Tipo:** Variable cualitativa categórica binaria
- **Describe:** Indica si el pasajero sobrevivió o no
- **Significado:** Supervivencia

**Valores:**
- `0` = No sobrevivió
- `1` = Sobrevivió

Es la **variable objetivo** del problema de clasificación.

### `Pclass`
- **Tipo:** Variable cualitativa ordinal
- **Describe:** Clase socioeconómica del pasajero
- **Significado:** Clase del pasaje

**Valores:**
- `1` = Clase alta
- `2` = Clase media
- `3` = Clase baja

Representa el nivel económico del pasajero.

### `Name`
- **Tipo:** Variable cualitativa nominal
- **Describe:** Nombre completo del pasajero
- **Significado:** Nombre

### `Sex`
- **Tipo:** Variable cualitativa nominal binaria
- **Describe:** Sexo del pasajero
- **Significado:** Sexo

**Valores:**
- `male` = hombre
- `female` = mujer

### `Age`
- **Tipo:** Variable cuantitativa continua
- **Describe:** Edad del pasajero en años
- **Significado:** Edad

### `SibSp`
- **Tipo:** Variable cuantitativa discreta
- **Describe:** Número de hermanos o cónyuges a bordo
- **Significado:** Hermanos / esposo(a) en el barco

### `Parch`
- **Tipo:** Variable cuantitativa discreta
- **Describe:** Número de padres o hijos a bordo
- **Significado:** Padres / hijos en el barco

### `Ticket`
- **Tipo:** Variable cualitativa nominal
- **Describe:** Número o código del tiquete
- **Significado:** Número del pasaje

No suele tener valor predictivo directo.

### `Fare`
- **Tipo:** Variable cuantitativa continua
- **Describe:** Precio pagado por el pasaje
- **Significado:** Tarifa del tiquete

Representa el nivel económico.

### `Cabin`
- **Tipo:** Variable cualitativa nominal
- **Describe:** Número o ubicación de la cabina
- **Significado:** Cabina

Tiene muchos valores faltantes.

### `Embarked`
- **Tipo:** Variable cualitativa nominal
- **Describe:** Puerto donde el pasajero abordó
- **Significado:** Puerto de embarque

**Valores:**
- `C` = Cherbourg
- `Q` = Queenstown
- `S` = Southampton

## G) Número de observaciones
**891 pasajeros**

## H) Número de variables
**12 variables**, incluida la variable respuesta

## I) Posibles hipótesis

### Hipótesis general
- **H₀:** Las características de los pasajeros no influyen en la probabilidad de supervivencia.
- **H₁:** Al menos una característica del pasajero influye en la supervivencia.

### Hipótesis específicas

#### Sexo
- **H₀:** La probabilidad de supervivencia es igual para hombres y mujeres.
- **H₁:** Las mujeres tienen mayor probabilidad de supervivencia.

#### Clase del pasaje
- **H₀:** La clase del pasaje no influye en la supervivencia.
- **H₁:** Los pasajeros de clases altas tienen mayor probabilidad de sobrevivir.

#### Edad
- **H₀:** La edad no influye en la supervivencia.
- **H₁:** Los pasajeros más jóvenes tienen mayor probabilidad de sobrevivir.

#### Tarifa
- **H₀:** El precio del tiquete no influye en la supervivencia.
- **H₁:** Los pasajeros que pagaron más tienen mayor probabilidad de sobrevivir.

## Posibles técnicas estadísticas para resolver las hipótesis
Como es un problema de clasificación, se pueden usar las siguientes técnicas:

- **Estadística descriptiva**, con tablas de frecuencia y tasas de supervivencia por grupo
- **Prueba Chi-cuadrado**, para variables categóricas como sexo vs. supervivencia o clase vs. supervivencia
- **Prueba t** o **Mann-Whitney**, para comparar edad promedio o tarifa promedio entre sobrevivientes y no sobrevivientes
- **Regresión logística**, para estimar la probabilidad de sobrevivir, evaluar la significancia de las variables y construir un modelo predictivo