# Documentación base de datos de clasificación

## a) Nombre de la base de datos
Chronic Kidney Disease / dataset_clasificacion.csv

## b) Fuente (URL)
https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

## c) Descripción general del problema
Esta base de datos contiene información clínica y de laboratorio de pacientes, con el fin de identificar si una persona presenta o no enfermedad renal crónica. El problema consiste en clasificar a los pacientes en dos grupos: con enfermedad renal crónica o sin enfermedad renal crónica, a partir de variables médicas, físicas y de laboratorio.

## d) Objetivo del análisis
Construir un modelo de clasificación que permita predecir si un paciente presenta enfermedad renal crónica, usando la información clínica y de laboratorio disponible en la base de datos.

## e) Variable objetivo (variable respuesta)
class

## f) Diccionario de variables

- age: edad del paciente — Numérica
- bp: presión arterial — Numérica
- sg: gravedad específica de la orina — Numérica
- al: albúmina — Numérica
- su: azúcar en orina — Numérica
- rbc: glóbulos rojos — Categórica nominal
- pc: células de pus — Categórica nominal
- pcc: cúmulos de células de pus — Categórica nominal
- ba: bacterias — Categórica nominal
- bgr: glucosa aleatoria en sangre — Numérica
- bu: urea en sangre — Numérica
- sc: creatinina sérica — Numérica
- sod: sodio — Numérica
- pot: potasio — Numérica
- hemo: hemoglobina — Numérica
- pcv: volumen celular empaquetado — Numérica
- wbcc: conteo de glóbulos blancos — Numérica
- rbcc: conteo de glóbulos rojos — Numérica
- htn: hipertensión — Categórica nominal
- dm: diabetes mellitus — Categórica nominal
- cad: enfermedad arterial coronaria — Categórica nominal
- appet: apetito — Categórica nominal
- pe: edema pedal — Categórica nominal
- ane: anemia — Categórica nominal
- class: presencia de enfermedad renal crónica — Categórica nominal binaria

## g) Número de observaciones
400

## h) Número de variables
25

## i) Posibles hipótesis de estudio
1. Los pacientes con niveles altos de creatinina sérica y urea en sangre tienen mayor probabilidad de presentar enfermedad renal crónica.
2. La hipertensión y la diabetes mellitus están asociadas con la presencia de enfermedad renal crónica.
3. Los indicadores hematológicos, como hemoglobina, volumen celular empaquetado y conteo de glóbulos rojos, ayudan a predecir la clase del paciente.
4. La combinación de variables clínicas y de laboratorio permite clasificar adecuadamente a los pacientes entre CKD y no CKD.