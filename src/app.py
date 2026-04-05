import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, kruskal, pearsonr

st.set_page_config(page_title="Laboratorio 3", layout="wide")

st.title("Laboratorio 3 - Analítica de Datos")
st.subheader("Selección de base de datos")

opcion = st.selectbox(
    "Elige la base de datos",
    ["Clasificación", "Regresión"]
)

def cargar_datos(opcion):
    if opcion == "Clasificación":
        conn = sqlite3.connect("./database/clasificacion.db")
        df = pd.read_sql_query("SELECT * FROM clasificacion", conn)
        conn.close()
        return df
    else:
        conn = sqlite3.connect("./database/regresion.db")
        df = pd.read_sql_query("SELECT * FROM regresion", conn)
        conn.close()
        return df

df = cargar_datos(opcion)

# Corrección para regresión
if opcion == "Regresión" and "charges" in df.columns:
    df["charges"] = pd.to_numeric(df["charges"], errors="coerce")

st.write(f"**Base seleccionada:** {opcion}")

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

st.subheader("Dimensiones del dataset")
st.write(f"Filas: {df.shape[0]}")
st.write(f"Columnas: {df.shape[1]}")

st.subheader("Tipos de datos")
tipos = pd.DataFrame(df.dtypes, columns=["Tipo de dato"])
st.dataframe(tipos)

st.subheader("Estadísticas descriptivas")
st.dataframe(df.describe(include="all"))

# -------------------------
# Gráficos principales
# -------------------------
st.subheader("Gráficos principales")

variables_numericas = df.select_dtypes(include="number").columns.tolist()

if variables_numericas:
    variable_hist = st.selectbox(
        "Selecciona una variable numérica para histograma",
        variables_numericas
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[variable_hist].dropna(), bins=15)
    ax.set_title(f"Histograma de {variable_hist}")
    ax.set_xlabel(variable_hist)
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

if opcion == "Clasificación":
    if "class" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="class", data=df, ax=ax)
        ax.set_title("Distribución de la variable objetivo (class)")
        st.pyplot(fig)

    if all(col in df.columns for col in ["class", "hemo"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="class", y="hemo", data=df, ax=ax)
        ax.set_title("Hemoglobina por clase")
        st.pyplot(fig)

else:
    if all(col in df.columns for col in ["age", "charges"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x="age", y="charges", data=df, ax=ax)
        ax.set_title("Edad vs cargos médicos")
        st.pyplot(fig)

    if all(col in df.columns for col in ["smoker", "charges"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="smoker", y="charges", data=df, ax=ax)
        ax.set_title("Cargos médicos según hábito de fumar")
        st.pyplot(fig)

# -------------------------
# Correlaciones
# -------------------------
st.subheader("Correlaciones")

if len(variables_numericas) >= 2:
    corr = df[variables_numericas].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Matriz de correlación")
    st.pyplot(fig)
else:
    st.write("No hay suficientes variables numéricas para calcular correlaciones.")

# -------------------------
# Imputación de datos faltantes
# -------------------------
st.subheader("Imputación de datos faltantes")

faltantes_antes = df.isnull().sum()
st.write("Valores faltantes antes de la imputación:")
st.dataframe(faltantes_antes[faltantes_antes > 0])

df_imputado = df.copy()

variables_numericas = df_imputado.select_dtypes(include="number").columns.tolist()
variables_categoricas = df_imputado.select_dtypes(exclude="number").columns.tolist()

# Imputación simple
for col in variables_numericas:
    df_imputado[col] = df_imputado[col].fillna(df_imputado[col].mean())

for col in variables_categoricas:
    if df_imputado[col].isnull().sum() > 0:
        df_imputado[col] = df_imputado[col].fillna(df_imputado[col].mode()[0])

faltantes_despues = df_imputado.isnull().sum()
st.write("Valores faltantes después de la imputación:")
st.dataframe(faltantes_despues[faltantes_despues > 0])

st.write("Vista previa de los datos imputados:")
st.dataframe(df_imputado.head())

# -------------------------
# Pruebas estadísticas
# -------------------------
st.subheader("Pruebas estadísticas")

if opcion == "Clasificación":
    st.write("### Base de clasificación")

    # Chi-cuadrado: htn vs class
    if all(col in df_imputado.columns for col in ["htn", "class"]):
        tabla = pd.crosstab(df_imputado["htn"], df_imputado["class"])
        chi2, p, dof, expected = chi2_contingency(tabla)

        st.write("**Chi-cuadrado: htn vs class**")
        st.dataframe(tabla)
        st.write(f"Chi-cuadrado: {chi2:.4f}")
        st.write(f"p-valor: {p:.6e}")
        if p < 0.05:
            st.success("Existe asociación significativa entre htn y class.")
        else:
            st.info("No se encontró asociación significativa entre htn y class.")

    # Spearman: hemo vs sc
    if all(col in df_imputado.columns for col in ["hemo", "sc"]):
        subset = df_imputado[["hemo", "sc"]].dropna()
        rho, p = spearmanr(subset["hemo"], subset["sc"])

        st.write("**Spearman: hemo vs sc**")
        st.write(f"Coeficiente de Spearman: {rho:.4f}")
        st.write(f"p-valor: {p:.6e}")
        if p < 0.05:
            st.success("Existe dependencia significativa entre hemo y sc.")
        else:
            st.info("No se encontró dependencia significativa entre hemo y sc.")

    # Kruskal-Wallis: hemo según class
    if all(col in df_imputado.columns for col in ["hemo", "class"]):
        grupo_ckd = df_imputado[df_imputado["class"] == "ckd"]["hemo"].dropna()
        grupo_notckd = df_imputado[df_imputado["class"] == "notckd"]["hemo"].dropna()

        if len(grupo_ckd) > 0 and len(grupo_notckd) > 0:
            stat, p = kruskal(grupo_ckd, grupo_notckd)

            st.write("**Kruskal-Wallis: hemo según class**")
            st.write(f"Estadístico: {stat:.4f}")
            st.write(f"p-valor: {p:.6e}")
            if p < 0.05:
                st.success("Existen diferencias significativas de hemo entre ckd y notckd.")
            else:
                st.info("No se encontraron diferencias significativas de hemo entre ckd y notckd.")

else:
    st.write("### Base de regresión")

    # Chi-cuadrado: smoker vs sex
    if all(col in df_imputado.columns for col in ["smoker", "sex"]):
        tabla = pd.crosstab(df_imputado["smoker"], df_imputado["sex"])
        chi2, p, dof, expected = chi2_contingency(tabla)

        st.write("**Chi-cuadrado: smoker vs sex**")
        st.dataframe(tabla)
        st.write(f"Chi-cuadrado: {chi2:.4f}")
        st.write(f"p-valor: {p:.6e}")
        if p < 0.05:
            st.success("Existe asociación significativa entre smoker y sex.")
        else:
            st.info("No se encontró asociación significativa entre smoker y sex.")

    # Pearson: age vs charges
    if all(col in df_imputado.columns for col in ["age", "charges"]):
        subset = df_imputado[["age", "charges"]].dropna()
        r, p = pearsonr(subset["age"], subset["charges"])

        st.write("**Pearson: age vs charges**")
        st.write(f"Coeficiente de Pearson: {r:.4f}")
        st.write(f"p-valor: {p:.6e}")
        if p < 0.05:
            st.success("Existe dependencia lineal significativa entre age y charges.")
        else:
            st.info("No se encontró dependencia lineal significativa entre age y charges.")

    # Kruskal-Wallis: charges según smoker
    if all(col in df_imputado.columns for col in ["charges", "smoker"]):
        grupo_no = df_imputado[df_imputado["smoker"] == "no"]["charges"].dropna()
        grupo_yes = df_imputado[df_imputado["smoker"] == "yes"]["charges"].dropna()

        if len(grupo_no) > 0 and len(grupo_yes) > 0:
            stat, p = kruskal(grupo_no, grupo_yes)

            st.write("**Kruskal-Wallis: charges según smoker**")
            st.write(f"Estadístico: {stat:.4f}")
            st.write(f"p-valor: {p:.6e}")
            if p < 0.05:
                st.success("Existen diferencias significativas de charges entre fumadores y no fumadores.")
            else:
                st.info("No se encontraron diferencias significativas de charges entre fumadores y no fumadores.")