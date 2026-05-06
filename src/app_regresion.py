import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="Laboratorio 4 - Modelación - Regresión",
    layout="wide"
)

st.title("Laboratorio 4 - Modelación - Regresión")
st.subheader("Predicción de cargos médicos en seguros")


# =====================================================
# RUTAS DEL PROYECTO
# =====================================================

project_root = Path(__file__).resolve().parents[1]

db_path = project_root / "database" / "regresion.db"

model_path = project_root / "models" / "model_regression.joblib"
features_path = project_root / "models" / "features_regression.joblib"

reports_dir = project_root / "reports"

tabla_comparacion_path = reports_dir / "tabla_comparacion_modelos.csv"
comparacion_hiper_path = reports_dir / "comparacion_hiperparametros.csv"
resumen_modelo_final_path = reports_dir / "resumen_modelo_final.csv"
comparacion_cv_test_path = reports_dir / "comparacion_cv_test.csv"


# =====================================================
# CARGA DE DATOS, MODELO Y REPORTES
# =====================================================

@st.cache_data
def load_data():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM regresion;", conn)
    conn.close()
    return df


@st.cache_resource
def load_model():
    return joblib.load(model_path)


@st.cache_data
def load_features():
    return joblib.load(features_path)


@st.cache_data
def load_reports():
    reports = {}

    if tabla_comparacion_path.exists():
        reports["tabla_comparacion"] = pd.read_csv(tabla_comparacion_path)

    if comparacion_hiper_path.exists():
        reports["comparacion_hiper"] = pd.read_csv(comparacion_hiper_path)

    if resumen_modelo_final_path.exists():
        reports["resumen_modelo_final"] = pd.read_csv(resumen_modelo_final_path)

    if comparacion_cv_test_path.exists():
        reports["comparacion_cv_test"] = pd.read_csv(comparacion_cv_test_path)

    return reports


df = load_data()
model = load_model()
features = load_features()
reports = load_reports()


# =====================================================
# PREPARACIÓN PARA EVALUACIÓN EN TEST
# =====================================================

X = df[features]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

y_pred_test = model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

residuos = y_test - y_pred_test


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navegación")

seccion = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Contexto del problema",
        "Datos y preparación",
        "Modelos entrenados",
        "Validación cruzada",
        "Comparación de modelos",
        "Ajuste de hiperparámetros",
        "Evaluación final",
        "Interpretabilidad",
        "Análisis de residuos",
        "Predicción interactiva",
        "Conclusiones"
    ]
)


# =====================================================
# CONTEXTO DEL PROBLEMA
# =====================================================

if seccion == "Contexto del problema":

    st.header("Contexto del problema")

    st.write(
        """
        El objetivo del laboratorio es construir un pipeline completo de modelación
        para un problema de regresión.

        En este caso, se trabaja con una base de datos de seguros médicos.
        La variable objetivo es **charges**, que representa los cargos médicos
        asociados a una persona asegurada.
        """
    )

    st.markdown("### Tipo de problema")

    st.write(
        """
        Este es un problema de **aprendizaje supervisado de regresión**, porque se busca
        predecir una variable numérica continua.

        Además, se trata de una regresión múltiple, ya que se utilizan varias variables
        predictoras para estimar los cargos médicos.
        """
    )

    st.markdown("### Variable objetivo")

    st.info("Variable objetivo: `charges`")

    st.markdown("### Variables predictoras")

    st.write(
        """
        - `age`: edad de la persona.
        - `sex`: sexo.
        - `bmi`: índice de masa corporal.
        - `children`: número de hijos o dependientes.
        - `smoker`: indica si la persona fuma.
        - `region`: región de residencia.
        """
    )


# =====================================================
# DATOS Y PREPARACIÓN
# =====================================================

elif seccion == "Datos y preparación":

    st.header("Datos y preparación")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Registros", df.shape[0])

    with col2:
        st.metric("Variables", df.shape[1])

    with col3:
        st.metric("Train/Test", "70% / 30%")

    st.markdown("### Primeras filas de la base")

    st.dataframe(df.head(10))

    st.markdown("### Tipos de variables")

    tipos = pd.DataFrame({
        "Variable": df.columns,
        "Tipo de dato": df.dtypes.astype(str).values
    })

    st.dataframe(tipos)

    st.markdown("### Variables numéricas y categóricas")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Variables numéricas**")
        st.write(["age", "bmi", "children"])

    with col2:
        st.write("**Variables categóricas**")
        st.write(["sex", "smoker", "region"])

    st.markdown("### Preprocesamiento")

    st.write(
        """
        El preprocesamiento se realizó dentro de un `Pipeline`, usando:

        - `StandardScaler` para variables numéricas.
        - `OneHotEncoder` para variables categóricas.

        Esto evita fuga de información, porque las transformaciones se ajustan
        únicamente con los datos de entrenamiento.
        """
    )

    st.markdown("### Matriz de correlación")

    numeric_cols = ["age", "bmi", "children", "charges"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr)

    ax.set_xticks(np.arange(len(numeric_cols)))
    ax.set_yticks(np.arange(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols)
    ax.set_yticklabels(numeric_cols)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    ax.set_title("Matriz de correlación")
    fig.colorbar(im)

    st.pyplot(fig)


# =====================================================
# MODELOS ENTRENADOS
# =====================================================

elif seccion == "Modelos entrenados":

    st.header("Modelos entrenados")

    st.write(
        """
        Se entrenaron diferentes modelos de regresión para comparar su desempeño.
        Todos los modelos fueron implementados dentro de un pipeline con el mismo
        preprocesamiento.
        """
    )

    modelos = pd.DataFrame({
        "Modelo": [
            "Regresión Lineal",
            "Ridge",
            "LASSO",
            "Árbol de Decisión",
            "Random Forest",
            "XGBoost",
            "LightGBM"
        ],
        "Tipo": [
            "Lineal",
            "Lineal regularizado",
            "Lineal regularizado",
            "Árbol",
            "Ensamble de árboles",
            "Boosting",
            "Boosting"
        ]
    })

    st.dataframe(modelos)

    st.markdown("### Nota sobre regresión isotónica")

    st.write(
        """
        La regresión isotónica no se incluyó como modelo principal porque este problema
        utiliza múltiples variables predictoras, tanto numéricas como categóricas.
        La regresión isotónica suele aplicarse cuando se desea modelar una relación
        monótona con una única variable explicativa.
        """
    )


# =====================================================
# VALIDACIÓN CRUZADA
# =====================================================

elif seccion == "Validación cruzada":

    st.header("Validación cruzada")

    st.write(
        """
        Para comparar los modelos se aplicó validación cruzada K-Fold con k = 5
        sobre el conjunto de entrenamiento.

        Las métricas utilizadas fueron:

        - MAE
        - RMSE
        - R²
        """
    )

    if "tabla_comparacion" in reports:

        tabla = reports["tabla_comparacion"]

        st.markdown("### Resultados de validación cruzada")

        st.dataframe(tabla)

        if "RMSE std" in tabla.columns:
            modelo_estable = tabla.sort_values(by="RMSE std", ascending=True).iloc[0]
            modelo_varianza = tabla.sort_values(by="RMSE std", ascending=False).iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Modelo más estable", modelo_estable["Modelo"])

            with col2:
                st.metric("Mayor varianza", modelo_varianza["Modelo"])

        st.write(
            """
            Un modelo más estable es aquel que presenta menor desviación estándar
            en validación cruzada. Una mayor desviación estándar indica mayor variabilidad
            entre los folds.
            """
        )

    else:
        st.warning("No se encontró el archivo tabla_comparacion_modelos.csv en reports/.")


# =====================================================
# COMPARACIÓN DE MODELOS
# =====================================================

elif seccion == "Comparación de modelos":

    st.header("Comparación de modelos")

    if "tabla_comparacion" in reports:

        tabla = reports["tabla_comparacion"].copy()

        st.markdown("### Tabla comparativa")

        st.dataframe(tabla)

        st.markdown("### Comparación normalizada de MAE, RMSE y R²")

        def normalizar_serie(serie):
            if serie.max() == serie.min():
                return serie * 0
            return (serie - serie.min()) / (serie.max() - serie.min())

        tabla_metricas = tabla.copy()

        tabla_metricas["MAE normalizado"] = 1 - normalizar_serie(tabla_metricas["MAE (CV)"])
        tabla_metricas["RMSE normalizado"] = 1 - normalizar_serie(tabla_metricas["RMSE (CV)"])
        tabla_metricas["R2 normalizado"] = normalizar_serie(tabla_metricas["R2 (CV)"])

        metricas_largas = tabla_metricas.melt(
            id_vars="Modelo",
            value_vars=["MAE normalizado", "RMSE normalizado", "R2 normalizado"],
            var_name="Métrica",
            value_name="Valor normalizado"
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        modelos = metricas_largas["Modelo"].unique()
        metricas = metricas_largas["Métrica"].unique()

        x = np.arange(len(modelos))
        width = 0.25

        for i, metrica in enumerate(metricas):
            valores = metricas_largas[metricas_largas["Métrica"] == metrica]["Valor normalizado"]
            ax.bar(x + i * width, valores, width, label=metrica)

        ax.set_xticks(x + width)
        ax.set_xticklabels(modelos, rotation=45, ha="right")
        ax.set_ylabel("Valor normalizado")
        ax.set_title("Comparación normalizada de métricas por modelo")
        ax.legend()
        ax.set_ylim(0, 1.05)

        st.pyplot(fig)

        mejor_modelo = tabla.sort_values(by="RMSE (CV)", ascending=True).iloc[0]

        st.success(
            f"Según el RMSE promedio en validación cruzada, el mejor modelo fue: {mejor_modelo['Modelo']}."
        )

        st.write(
            """
            En esta gráfica, las métricas fueron normalizadas para poder compararlas
            en una misma escala. Para MAE y RMSE, como valores más bajos son mejores,
            se invirtió la escala para que barras más altas indiquen mejor desempeño.
            """
        )

    else:
        st.warning("No se encontró el archivo tabla_comparacion_modelos.csv en reports/.")


# =====================================================
# AJUSTE DE HIPERPARÁMETROS
# =====================================================

elif seccion == "Ajuste de hiperparámetros":

    st.header("Ajuste de hiperparámetros")

    st.write(
        """
        Se aplicó búsqueda de hiperparámetros usando validación cruzada.
        El objetivo fue mejorar el desempeño de modelos seleccionados comparando
        las métricas antes y después del ajuste.
        """
    )

    if "comparacion_hiper" in reports:

        tabla_hiper = reports["comparacion_hiper"]

        st.markdown("### Comparación antes vs después del ajuste")

        st.dataframe(tabla_hiper)

        st.markdown("### Cambio en RMSE antes y después")

        fig, ax = plt.subplots(figsize=(9, 5))

        x = np.arange(len(tabla_hiper["Modelo"]))
        width = 0.35

        ax.bar(x - width / 2, tabla_hiper["RMSE antes"], width, label="RMSE antes")
        ax.bar(x + width / 2, tabla_hiper["RMSE después"], width, label="RMSE después")

        ax.set_xticks(x)
        ax.set_xticklabels(tabla_hiper["Modelo"], rotation=45, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE antes vs después del ajuste")
        ax.legend()

        st.pyplot(fig)

        st.write(
            """
            Un modelo mejora después del ajuste si presenta menor MAE, menor RMSE
            y mayor R².
            """
        )

    else:
        st.warning("No se encontró el archivo comparacion_hiperparametros.csv en reports/.")


# =====================================================
# EVALUACIÓN FINAL
# =====================================================

elif seccion == "Evaluación final":

    st.header("Evaluación final del modelo")

    st.write(
        """
        El mejor modelo ajustado fue evaluado sobre el conjunto de prueba.
        Este conjunto no fue usado durante el entrenamiento ni durante la validación cruzada.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE en test", f"{test_mae:,.2f}")

    with col2:
        st.metric("RMSE en test", f"{test_rmse:,.2f}")

    with col3:
        st.metric("R² en test", f"{test_r2:.4f}")

    if "comparacion_cv_test" in reports:

        st.markdown("### Comparación CV vs Test")

        st.dataframe(reports["comparacion_cv_test"])

    st.markdown("### Valores reales vs predichos")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(y_test, y_pred_test, alpha=0.6)

    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Valores reales")
    ax.set_ylabel("Valores predichos")
    ax.set_title("Valores reales vs predichos")

    st.pyplot(fig)

    st.write(
        f"""
        El R² en test fue de **{test_r2:.4f}**, lo que indica que el modelo explica
        aproximadamente el **{test_r2 * 100:.2f}%** de la variabilidad de los cargos médicos
        en datos no vistos.
        """
    )


# =====================================================
# INTERPRETABILIDAD
# =====================================================

elif seccion == "Interpretabilidad":

    st.header("Interpretabilidad del modelo")

    st.write(
        """
        La interpretabilidad permite analizar qué variables tienen mayor peso
        en las predicciones del modelo.
        """
    )

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):

        importancias = estimator.feature_importances_

        importance_df = pd.DataFrame({
            "Variable": feature_names,
            "Importancia": importancias
        }).sort_values(by="Importancia", ascending=False)

        st.markdown("### Importancia de variables")

        st.dataframe(importance_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df["Variable"], importance_df["Importancia"])
        ax.set_xlabel("Importancia")
        ax.set_ylabel("Variable")
        ax.set_title("Importancia de variables del modelo final")
        ax.invert_yaxis()

        st.pyplot(fig)

        st.write(
            """
            Las variables con mayor importancia son las que más aportan a la predicción
            de los cargos médicos dentro del modelo final.
            """
        )

    elif hasattr(estimator, "coef_"):

        coeficientes = estimator.coef_

        coef_df = pd.DataFrame({
            "Variable": feature_names,
            "Coeficiente": coeficientes
        })

        coef_df["Abs_Coeficiente"] = np.abs(coef_df["Coeficiente"])

        coef_df = coef_df.sort_values(by="Abs_Coeficiente", ascending=False)

        st.markdown("### Coeficientes del modelo lineal")

        st.dataframe(coef_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(coef_df["Variable"], coef_df["Coeficiente"])
        ax.set_xlabel("Coeficiente")
        ax.set_ylabel("Variable")
        ax.set_title("Coeficientes del modelo final")
        ax.invert_yaxis()

        st.pyplot(fig)

    else:
        st.warning("El modelo final no tiene coeficientes ni importancias disponibles.")


# =====================================================
# ANÁLISIS DE RESIDUOS
# =====================================================

elif seccion == "Análisis de residuos":

    st.header("Análisis de residuos")

    st.write(
        """
        Los residuos se calculan como:

        **residuo = valor real - valor predicho**

        Este análisis permite revisar si los errores presentan patrones no aleatorios
        o problemas de homocedasticidad.
        """
    )

    st.markdown("### Residuos vs predicciones")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(y_pred_test, residuos, alpha=0.6)
    ax.axhline(y=0, linestyle="--")

    ax.set_xlabel("Predicciones")
    ax.set_ylabel("Residuos")
    ax.set_title("Residuos vs predicciones")

    st.pyplot(fig)

    st.write(
        """
        Si los residuos se distribuyen alrededor de cero sin un patrón claro,
        el comportamiento del modelo es más adecuado. Si aparece una forma de embudo
        o una curva, puede existir heterocedasticidad o un patrón no capturado por el modelo.
        """
    )

    st.markdown("### Distribución de residuos")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(residuos, bins=30)
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de residuos")

    st.pyplot(fig)

    resumen_residuos = pd.DataFrame({
        "Métrica": [
            "Media de residuos",
            "Desviación estándar",
            "Residuo mínimo",
            "Residuo máximo"
        ],
        "Valor": [
            residuos.mean(),
            residuos.std(),
            residuos.min(),
            residuos.max()
        ]
    })

    st.markdown("### Resumen de residuos")

    st.dataframe(resumen_residuos)


# =====================================================
# PREDICCIÓN INTERACTIVA
# =====================================================

elif seccion == "Predicción interactiva":

    st.header("Predicción interactiva")

    st.write(
        """
        En esta sección se puede ingresar la información de una persona asegurada
        y el modelo estima sus cargos médicos.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Edad", min_value=18, max_value=64, value=35)
        bmi = st.number_input("Índice de masa corporal (BMI)", min_value=10.0, max_value=60.0, value=28.0)
        children = st.slider("Número de hijos/dependientes", min_value=0, max_value=5, value=1)

    with col2:
        sex = st.selectbox("Sexo", ["female", "male"])
        smoker = st.selectbox("¿Fuma?", ["no", "yes"])
        region = st.selectbox(
            "Región",
            ["southwest", "southeast", "northwest", "northeast"]
        )

    nuevo_registro = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    st.markdown("### Datos ingresados")

    st.dataframe(nuevo_registro)

    if st.button("Predecir cargos médicos"):

        prediccion = model.predict(nuevo_registro)[0]

        st.success(f"Predicción estimada de cargos médicos: {prediccion:,.2f}")

        st.write(
            """
            Esta predicción es una estimación generada por el modelo final entrenado.
            No representa un valor real obligatorio, sino una aproximación basada en los patrones encontrados en los datos.
            """
        )


# =====================================================
# CONCLUSIONES
# =====================================================

elif seccion == "Conclusiones":

    st.header("Conclusiones")

    st.write(
        """
        En este laboratorio se desarrolló un pipeline completo para un problema de regresión.

        Se compararon múltiples modelos mediante validación cruzada, se ajustaron
        hiperparámetros, se evaluó el mejor modelo en test y se analizó la interpretabilidad
        y los residuos.
        """
    )

    st.markdown("### Archivos generados")

    st.write(
        """
        - `models/model_regression.joblib`: contiene el modelo final entrenado.
        - `models/features_regression.joblib`: contiene la lista de variables predictoras utilizadas.
        """
    )

    st.success(
        "El modelo final quedó guardado correctamente y puede reutilizarse para hacer nuevas predicciones."
    )