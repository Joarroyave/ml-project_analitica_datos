# Documentación base de datos de clasificación

## a) Nombre de la base de datos
Análisis de sentimientos / dataset_clasificacion.csv

## b) Fuente (URL)
(https://www.kaggle.com/datasets/vanesamizrahi/olist-sentiment-analysis-huggingface)

## c) Descripción general del problema
Esta base de datos contiene reseñas de clientes y variables asociadas al análisis de sentimiento. El problema consiste en clasificar la polaridad del sentimiento de una reseña a partir del comentario escrito y otras variables relacionadas.

## d) Objetivo del análisis
Construir un modelo de clasificación que permita predecir la polaridad del sentimiento de una reseña usando la información disponible en la base de datos.

## e) Variable objetivo (variable respuesta)
sentiment_polarity

## f) Diccionario de variables

| Nombre de la variable | Descripción | Tipo de variable |
|---|---|---|
| review_id | Identificador de la reseña | Categórica nominal |
| order_id | Identificador del pedido | Categórica nominal |
| review_score | Calificación del cliente | Numérica discreta |
| review_comment_message | Comentario escrito por el cliente | Categórica nominal |
| sentiment_label_raw | Etiqueta textual del sentimiento | Categórica ordinal |
| sentiment_score_raw | Puntaje del análisis de sentimiento | Numérica continua |
| sentiment_stars | Número de estrellas asociado al sentimiento | Numérica discreta |
| sentiment_polarity | Polaridad del sentimiento | Categórica ordinal |

## g) Número de observaciones
44777

## h) Número de variables
8

## i) Posibles hipótesis de estudio
1. Las reseñas con mayor calificación tienden a tener polaridad positiva.
2. El contenido del comentario permite predecir la polaridad del sentimiento.
3. El puntaje de sentimiento está asociado con la clase final de polaridad.
4. Las estrellas asignadas guardan relación con la polaridad del sentimiento.