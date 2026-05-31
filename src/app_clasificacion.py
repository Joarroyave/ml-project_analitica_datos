import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------

st.set_page_config(
    page_title="Modelación de Clasificación",
    page_icon="🩺",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "database" / "clasificacion.db"
MODEL_PATH = PROJECT_ROOT / "models" / "model_classification.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "features_classification.joblib"


# ---------------------------------------------------------
# Carga de datos y modelo
# ---------------------------------------------------------

@st.cache_data
def cargar_datos():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM clasificacion;", conn)
    conn.close()
    return df


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def cargar_features():
    return joblib.load(FEATURES_PATH)


df = cargar_datos()
modelo_final = cargar_modelo()
features_info = cargar_features()


# ---------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------

target = "class"

X = df.drop(columns=[target])
y = df[target].map({
    "notckd": 0,
    "ckd": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Predicciones del modelo final guardado: Regresión Logística base
y_pred = modelo_final.predict(X_test)
y_pred_proba = modelo_final.predict_proba(X_test)[:, 1]

# Métricas finales
accuracy_test = accuracy_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred)
recall_test = recall_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred)
auc_test = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()


# ---------------------------------------------------------
# Tablas principales del notebook
# ---------------------------------------------------------

tabla_cv = pd.DataFrame({
    "Modelo": [
        "Regresión Logística",
        "Árbol de Clasificación",
        "Random Forest",
        "XGBoost",
        "Balanced Random Forest"
    ],
    "Accuracy (CV)": [
        "0.9929 ± 0.0087",
        "0.9607 ± 0.0286",
        "0.9893 ± 0.0143",
        "0.9893 ± 0.0087",
        "0.9929 ± 0.0143"
    ],
    "Precision (CV)": [
        "1.0000 ± 0.0000",
        "0.9825 ± 0.0234",
        "0.9944 ± 0.0111",
        "0.9944 ± 0.0111",
        "1.0000 ± 0.0000"
    ],
    "Recall (CV)": [
        "0.9886 ± 0.0140",
        "0.9543 ± 0.0291",
        "0.9886 ± 0.0229",
        "0.9886 ± 0.0140",
        "0.9886 ± 0.0229"
    ],
    "F1-Score (CV)": [
        "0.9942 ± 0.0071",
        "0.9680 ± 0.0232",
        "0.9913 ± 0.0117",
        "0.9914 ± 0.0070",
        "0.9941 ± 0.0118"
    ],
    "AUC (CV)": [
        "0.9989 ± 0.0022",
        "0.9629 ± 0.0296",
        "1.0000 ± 0.0000",
        "0.9995 ± 0.0011",
        "1.0000 ± 0.0000"
    ]
})

tabla_ajuste = pd.DataFrame({
    "Modelo": [
        "Regresión Logística - Antes",
        "Regresión Logística - Después",
        "Balanced Random Forest - Antes",
        "Balanced Random Forest - Después"
    ],
    "Accuracy (CV)": [
        "0.9929 ± 0.0087",
        "0.6464 ± 0.0262",
        "0.9929 ± 0.0143",
        "0.9893 ± 0.0143"
    ],
    "Precision (CV)": [
        "1.0000 ± 0.0000",
        "0.6392 ± 0.0177",
        "1.0000 ± 0.0000",
        "0.9944 ± 0.0111"
    ],
    "Recall (CV)": [
        "0.9886 ± 0.0140",
        "1.0000 ± 0.0000",
        "0.9886 ± 0.0229",
        "0.9886 ± 0.0229"
    ],
    "F1-Score (CV)": [
        "0.9942 ± 0.0071",
        "0.7797 ± 0.0130",
        "0.9941 ± 0.0118",
        "0.9913 ± 0.0117"
    ],
    "AUC (CV)": [
        "0.9989 ± 0.0022",
        "0.9970 ± 0.0039",
        "1.0000 ± 0.0000",
        "1.0000 ± 0.0000"
    ]
})

tabla_logistica_vs_xgboost = pd.DataFrame({
    "Modelo": ["Regresión Logística base", "XGBoost base"],
    "Accuracy": [0.9917, 1.0000],
    "Precision": [1.0000, 1.0000],
    "Recall": [0.9867, 1.0000],
    "F1-Score": [0.9933, 1.0000],
    "AUC": [0.9997, 1.0000],
    "Falsos positivos": [0, 0],
    "Falsos negativos": [1, 0]
})


# ---------------------------------------------------------
# Función auxiliar para matrices de confusión
# ---------------------------------------------------------

def graficar_matriz_confusion(matriz, titulo):
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.imshow(matriz)

    ax.set_title(titulo)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["notckd", "ckd"])
    ax.set_yticklabels(["notckd", "ckd"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matriz[i, j], ha="center", va="center", fontsize=12)

    return fig


# ---------------------------------------------------------
# Título general
# ---------------------------------------------------------

st.title("Laboratorio 5 - Modelación de Clasificación")
st.markdown(
    """
    **Problema:** clasificación binaria para predecir si un paciente presenta enfermedad renal crónica.

    **Modelo final seleccionado:** Regresión Logística base.

    La selección no se realizó únicamente con Accuracy, sino considerando Recall, F1-Score, AUC, matriz de confusión, interpretabilidad y el tipo de error más crítico.
    """
)


# ---------------------------------------------------------
# Menú lateral
# ---------------------------------------------------------

seccion = st.sidebar.radio(
    "Contenido de la exposición",
    [
        "1. Contexto y datos",
        "2. Exploración inicial",
        "3. Pipeline y modelos",
        "4. Validación cruzada",
        "5. Ajuste de hiperparámetros",
        "6. Evaluación final",
        "7. Interpretabilidad",
        "8. Conclusión"
    ]
)


# ---------------------------------------------------------
# 1. Contexto y datos
# ---------------------------------------------------------

if seccion == "1. Contexto y datos":
    st.header("1. Contexto del problema")

    col1, col2, col3 = st.columns(3)

    col1.metric("Registros", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Variable objetivo", "class")

    st.markdown(
        """
        La variable objetivo es `class`, con dos categorías:

        - `ckd`: paciente con enfermedad renal crónica.
        - `notckd`: paciente sin enfermedad renal crónica.

        La clase positiva fue codificada como:

        - `ckd = 1`
        - `notckd = 0`
        """
    )

    st.subheader("Primeras filas de la base")
    st.dataframe(df.head(), use_container_width=True)


# ---------------------------------------------------------
# 2. Exploración inicial
# ---------------------------------------------------------

elif seccion == "2. Exploración inicial":
    st.header("2. Exploración inicial")

    st.subheader("Distribución de clases")

    distribucion = df["class"].value_counts().reset_index()
    distribucion.columns = ["Clase", "Cantidad"]
    distribucion["Porcentaje"] = round(distribucion["Cantidad"] / len(df) * 100, 2)

    st.dataframe(distribucion, use_container_width=True)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(distribucion["Clase"], distribucion["Cantidad"])
    ax.set_title("Distribución de la variable objetivo")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)

    st.markdown(
        """
        La base presenta un **desbalance moderado**:

        - `ckd`: 62.5%
        - `notckd`: 37.5%

        Por esta razón, Accuracy se reporta, pero no se usa como único criterio de selección.
        """
    )

    st.subheader("Valores faltantes")

    faltantes = pd.DataFrame({
        "Variable": df.isnull().sum().index,
        "Faltantes": df.isnull().sum().values,
        "Porcentaje": (df.isnull().mean().values * 100).round(2)
    }).sort_values(by="Faltantes", ascending=False)

    st.dataframe(faltantes, use_container_width=True)

    top_faltantes = faltantes[faltantes["Faltantes"] > 0].head(10)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(top_faltantes["Variable"], top_faltantes["Faltantes"])
    ax.set_title("Variables con más valores faltantes")
    ax.set_xlabel("Cantidad de faltantes")
    ax.invert_yaxis()
    st.pyplot(fig)


# ---------------------------------------------------------
# 3. Pipeline y modelos
# ---------------------------------------------------------

elif seccion == "3. Pipeline y modelos":
    st.header("3. Pipeline y modelos")

    st.markdown(
        """
        El preprocesamiento se realizó dentro de un `Pipeline` para evitar fuga de información.

        **Variables numéricas:**

        - Imputación con `KNNImputer(n_neighbors=5)`.
        - `StandardScaler` antes de KNN porque KNN usa distancias.
        - En Regresión Logística también se mantiene escalamiento por sensibilidad a la escala.

        **Variables categóricas:**

        - Imputación con la categoría más frecuente.
        - Codificación con `OneHotEncoder`.

        **Modelos entrenados:**

        - Regresión Logística.
        - Árbol de Clasificación.
        - Random Forest.
        - XGBoost.
        - Balanced Random Forest.
        """
    )

    st.subheader("Variables numéricas")
    st.write(features_info["numeric_features"])

    st.subheader("Variables categóricas")
    st.write(features_info["categorical_features"])

    st.info(
        "Balanced Random Forest se incluyó como alternativa de ensamble basada en árboles para considerar el desbalance de clases."
    )


# ---------------------------------------------------------
# 4. Validación cruzada
# ---------------------------------------------------------

elif seccion == "4. Validación cruzada":
    st.header("4. Validación cruzada")

    st.markdown(
        """
        Se aplicó `StratifiedKFold` con 5 particiones, conservando la proporción de clases en cada fold.

        Las métricas evaluadas fueron Accuracy, Precision, Recall, F1-Score y AUC.
        """
    )

    st.dataframe(tabla_cv, use_container_width=True)

    st.markdown(
        """
        **Lectura principal:**

        - Regresión Logística y Balanced Random Forest presentaron los mejores resultados generales.
        - El Árbol de Clasificación fue el modelo con menor desempeño relativo.
        - XGBoost también presentó resultados muy altos y luego se revisó como alternativa frente al error crítico.
        - Se prioriza Recall y F1-Score porque el problema está relacionado con detección de enfermedad.
        """
    )


# ---------------------------------------------------------
# 5. Ajuste de hiperparámetros
# ---------------------------------------------------------

elif seccion == "5. Ajuste de hiperparámetros":
    st.header("5. Ajuste de hiperparámetros")

    st.markdown(
        """
        Se aplicó `RandomizedSearchCV` a dos modelos:

        - Regresión Logística, como modelo lineal.
        - Balanced Random Forest, como modelo basado en árboles orientado al desbalance.

        La métrica principal para seleccionar hiperparámetros fue **Recall**, porque el error más crítico es el falso negativo.
        """
    )

    st.dataframe(tabla_ajuste, use_container_width=True)

    st.markdown(
        """
        **Conclusión del ajuste:**

        - La Regresión Logística ajustada alcanzó Recall de 1.0000, pero perdió mucho en Precision, Accuracy y F1-Score.
        - Balanced Random Forest ajustado no mejoró frente a su versión base.
        - Por tanto, los modelos base fueron más convenientes.
        - Esta comparación corresponde a los modelos ajustados. En la evaluación final se mantiene el modelo base seleccionado.
        """
    )


# ---------------------------------------------------------
# 6. Evaluación final
# ---------------------------------------------------------

elif seccion == "6. Evaluación final":
    st.header("6. Evaluación final en test")

    st.markdown(
        """
        El modelo final seleccionado fue la **Regresión Logística base**.

        Se evaluó sobre el conjunto de prueba, que no fue utilizado durante el entrenamiento ni la validación cruzada.
        """
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy_test:.4f}")
    col2.metric("Precision", f"{precision_test:.4f}")
    col3.metric("Recall", f"{recall_test:.4f}")
    col4.metric("F1-Score", f"{f1_test:.4f}")
    col5.metric("AUC", f"{auc_test:.4f}")

    st.subheader("Matriz de confusión del modelo final")

    matriz_logistica = pd.DataFrame(
        cm,
        index=["Real notckd", "Real ckd"],
        columns=["Predicho notckd", "Predicho ckd"]
    )

    st.dataframe(matriz_logistica, use_container_width=True)

    fig = graficar_matriz_confusion(cm, "Matriz de confusión - Regresión Logística")
    st.pyplot(fig)

    st.markdown(
        """
        La Regresión Logística clasificó correctamente 45 pacientes sin enfermedad renal crónica y 74 pacientes con enfermedad renal crónica.

        No presentó falsos positivos, pero sí presentó **1 falso negativo**, es decir, un paciente con enfermedad renal crónica fue clasificado como `notckd`.

        Este es el error más crítico del problema, porque podría implicar que un paciente enfermo no reciba seguimiento oportuno.
        """
    )

    st.subheader("Análisis complementario: ¿otro modelo mejora el falso negativo?")

    st.markdown(
        """
        Dado que la Regresión Logística presentó un falso negativo, se revisó si otro modelo reducía este error crítico.

        Para esta comparación se revisó **XGBoost base**, ya que en el análisis del notebook fue el modelo que no presentó falsos negativos en el conjunto de prueba.

        Esta comparación usa modelos base. Los modelos ajustados se revisaron en la sección anterior y no mejoraron frente a sus versiones base.
        """
    )

    st.dataframe(tabla_logistica_vs_xgboost, use_container_width=True)

    cm_logistica = np.array([[45, 0],
                             [1, 74]])

    cm_xgboost = np.array([[45, 0],
                           [0, 75]])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    matrices = {
        "Regresión Logística base": cm_logistica,
        "XGBoost base": cm_xgboost
    }

    for ax, (nombre_modelo, matriz) in zip(axes, matrices.items()):
        ax.imshow(matriz)
        ax.set_title(nombre_modelo)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["notckd", "ckd"])
        ax.set_yticklabels(["notckd", "ckd"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, matriz[i, j], ha="center", va="center", fontsize=12)

    plt.suptitle("Comparación de matrices de confusión en test", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(
        """
        En esta comparación, XGBoost no presentó falsos positivos ni falsos negativos en el conjunto de prueba.

        Esto significa que, para este conjunto de datos, XGBoost logró corregir el error crítico observado en la Regresión Logística.

        Sin embargo, la selección final del modelo no debe basarse únicamente en una matriz de confusión del conjunto de prueba. También se deben considerar la validación cruzada, la estabilidad, la interpretabilidad y la facilidad para explicar el modelo.

        Por eso, la Regresión Logística se mantiene como modelo final en el análisis principal, mientras que XGBoost queda como una alternativa fuerte si el objetivo principal fuera minimizar completamente los falsos negativos.
        """
    )

    st.subheader("Curva ROC del modelo final")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_test:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC - Regresión Logística")
    ax.legend()
    st.pyplot(fig)


# ---------------------------------------------------------
# 7. Interpretabilidad
# ---------------------------------------------------------

elif seccion == "7. Interpretabilidad":
    st.header("7. Interpretabilidad del modelo final")

    st.markdown(
        """
        Como el modelo final es una Regresión Logística, la interpretación se realiza mediante coeficientes.

        - Coeficiente positivo: aumenta la probabilidad de clasificar como `ckd`.
        - Coeficiente negativo: disminuye la probabilidad de clasificar como `ckd`.
        """
    )

    preprocessor_final = modelo_final.named_steps["preprocessor"]
    modelo_logistico = modelo_final.named_steps["model"]

    try:
        feature_names = preprocessor_final.get_feature_names_out()
    except Exception:
        ohe = preprocessor_final.named_transformers_["cat"].named_steps["onehot"]
        cat_features = ohe.get_feature_names_out(features_info["categorical_features"])

        num_features = np.array([
            f"num__{col}" for col in features_info["numeric_features"]
        ])

        cat_features = np.array([
            f"cat__{col}" for col in cat_features
        ])

        feature_names = np.concatenate([num_features, cat_features])

    coeficientes = modelo_logistico.coef_[0]

    tabla_coeficientes = pd.DataFrame({
        "Variable": feature_names,
        "Coeficiente": coeficientes,
        "Valor absoluto": np.abs(coeficientes)
    }).sort_values(by="Valor absoluto", ascending=False)

    st.dataframe(tabla_coeficientes.head(15), use_container_width=True)

    top_coeficientes = tabla_coeficientes.head(15).sort_values("Coeficiente")

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.barh(top_coeficientes["Variable"], top_coeficientes["Coeficiente"])
    ax.set_title("Principales coeficientes del modelo final")
    ax.set_xlabel("Coeficiente")
    ax.set_ylabel("Variable")
    st.pyplot(fig)

    st.markdown(
        """
        Las variables más influyentes fueron `sg`, `hemo`, `al`, `appet`, `pcv`, `sc`, `htn`, `dm`, `su`, `bp`, `rbcc` y `sod`.

        Variables como `al`, `sc`, `htn_yes`, `dm_yes`, `su` y `bp` aumentan la probabilidad de clasificar como `ckd`.

        Variables como `sg`, `hemo`, `pcv`, `rbcc` y `sod` disminuyen la probabilidad de clasificar como `ckd`.
        """
    )


# ---------------------------------------------------------
# 8. Conclusión
# ---------------------------------------------------------

elif seccion == "8. Conclusión":
    st.header("8. Conclusión del laboratorio")

    st.success("Modelo final seleccionado: Regresión Logística base")

    st.markdown(
        """
        La Regresión Logística base fue seleccionada como modelo final porque presentó:

        - Alto desempeño en validación cruzada.
        - Buen desempeño en el conjunto de prueba.
        - Recall alto, importante para reducir falsos negativos.
        - F1-Score alto, mostrando equilibrio entre Precision y Recall.
        - AUC alto, indicando buena capacidad discriminativa.
        - Mayor interpretabilidad frente a modelos más complejos.

        También se evaluó Balanced Random Forest como alternativa orientada al desbalance. Aunque tuvo resultados muy altos, no reemplazó al modelo final porque la Regresión Logística ofreció desempeño similar con mayor facilidad de interpretación.

        En el análisis complementario, XGBoost base eliminó el falso negativo en el conjunto de prueba y no empeoró las métricas en ese conjunto. Por eso se reconoce como una alternativa fuerte si el objetivo principal fuera minimizar completamente los falsos negativos.

        Aun así, se mantiene la Regresión Logística base como modelo final principal, porque la selección no depende solo de una matriz de confusión en test, sino también de la validación cruzada, la estabilidad, el equilibrio entre métricas y la interpretabilidad.
        """
    )