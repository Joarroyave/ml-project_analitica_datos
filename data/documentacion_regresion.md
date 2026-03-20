# Documentación base de datos de regresión

## A) Nombre de la base de datos
**Insurance Charges Dataset – Base de datos de cargos médicos**

## B) Fuente (URL)
https://media.geeksforgeeks.org/wp-content/uploads/20240522154112/insurance%5B1%5D.csv

## C) Descripción general del problema
La base de datos **Insurance Charges** contiene información sobre los costos médicos facturados por compañías aseguradoras de salud a diferentes personas.

El propósito del estudio es analizar cómo ciertas características demográficas y de salud influyen en el costo individual de los servicios médicos o del seguro de salud.

Este tipo de análisis es relevante en:

- economía de la salud
- análisis actuarial
- compañías aseguradoras
- estudios de riesgo médico

Los datos representan patrones realistas de facturación médica y son utilizados principalmente con fines educativos y de análisis estadístico.

Actualmente esta base es difundida en plataformas académicas y repositorios de ciencia de datos para el estudio de modelos de regresión y predicción de costos médicos.

## D) Objetivo del análisis
Construir un modelo estadístico que permita explicar y predecir el costo médico individual a partir de características personales y de salud.

Cada observación corresponde a una persona asegurada.

## E) Variable objetivo (variable respuesta)
La variable objetivo es **`charges`**.

- **Tipo:** Variable cuantitativa continua
- **Describe:** Costo total médico o de seguro generado por una persona
- **Unidad:** Dólares

Por tanto, el problema corresponde a un **modelo de regresión**.

## F) Diccionario de variables

### `age`
- **Tipo:** Cuantitativa discreta
- **Describe:** Edad de la persona

### `sex`
- **Tipo:** Cualitativa nominal
- **Describe:** Sexo

### `bmi`
- **Tipo:** Cuantitativa continua
- **Describe:** Índice de masa corporal

### `children`
- **Tipo:** Cuantitativa discreta
- **Describe:** Número de hijos dependientes

### `smoker`
- **Tipo:** Cualitativa nominal binaria
- **Describe:** Indica si la persona fuma

Valores:
- `yes` = fumador
- `no` = no fumador

### `region`
- **Tipo:** Cualitativa nominal
- **Describe:** Región de residencia

### `charges`
- **Tipo:** Cuantitativa continua
- **Describe:** Cargos médicos facturados

## G) Número de observaciones
La base contiene aproximadamente **1338 observaciones** (personas).

## H) Número de variables
La base contiene **7 variables**, incluyendo la variable respuesta.

## I) Posibles hipótesis

### Hipótesis general
- **H₀:** Las características personales y de salud no influyen en el costo médico.
- **H₁:** Al menos una característica influye en el costo médico.

### Hipótesis específicas

#### Hábito de fumar
- **H₀:** El hábito de fumar no influye significativamente en los cargos médicos individuales.
- **H₁:** El hábito de fumar influye significativamente en los cargos médicos individuales.

#### Edad
- **H₀:** La edad de las personas no influye significativamente en los cargos médicos.
- **H₁:** La edad influye significativamente en los cargos médicos.

#### Índice de masa corporal (BMI)
- **H₀:** El índice de masa corporal no influye significativamente en los cargos médicos.
- **H₁:** El índice de masa corporal influye significativamente en los cargos médicos.

#### Número de hijos
- **H₀:** El número de hijos no influye significativamente en los cargos médicos.
- **H₁:** El número de hijos influye significativamente en los cargos médicos.

## J) Técnicas estadísticas posibles
- Estadística descriptiva
- Diagramas de dispersión
- Correlación
- Prueba t de medias
- ANOVA
- Regresión lineal múltiple (técnica principal)
- Diagnóstico de residuos
- Transformación logarítmica

## Conclusión
Esta base permite analizar los factores que determinan el costo médico individual mediante modelos de regresión, siendo ampliamente utilizada en estudios predictivos de seguros de salud.