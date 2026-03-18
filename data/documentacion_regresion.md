# Documentación base de datos de regresión

## a) Nombre de la base de datos
California Housing Dataset / dataset_regresion.csv

## b) Fuente (URL)
https://media.geeksforgeeks.org/wp-content/uploads/20240319120216/housing.csv

## c) Descripción general del problema
Esta base de datos contiene información sobre viviendas, incluyendo variables relacionadas con ubicación, características de las casas, población e ingresos. El problema consiste en predecir el valor de una vivienda a partir de estas variables.

## d) Objetivo del análisis
Construir un modelo de regresión que permita estimar el valor de una vivienda usando las variables disponibles en la base de datos.

## e) Variable objetivo (variable respuesta)
median_house_value

## f) Diccionario de variables

| Nombre de la variable | Descripción | Tipo de variable |
|---|---|---|
| longitude | Longitud geográfica | Numérica continua |
| latitude | Latitud geográfica | Numérica continua |
| housing_median_age | Antigüedad mediana de las viviendas | Numérica continua |
| total_rooms | Número total de habitaciones | Numérica continua |
| total_bedrooms | Número total de dormitorios | Numérica continua |
| population | Población de la zona | Numérica continua |
| households | Número de hogares | Numérica continua |
| median_income | Ingreso mediano | Numérica continua |
| median_house_value | Valor mediano de la vivienda | Numérica continua |
| ocean_proximity | Cercanía al océano | Categórica nominal |

## g) Número de observaciones
20640

## h) Número de variables
10

## i) Posibles hipótesis de estudio
1. A mayor ingreso mediano, mayor valor de la vivienda.
2. La cercanía al océano influye en el valor de la vivienda.
3. La ubicación geográfica está asociada con diferencias en el precio.
4. El número de habitaciones y dormitorios se relaciona con el valor de la vivienda.