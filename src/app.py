import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, kruskal, pearsonr, f_oneway
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(df[variable_hist].dropna(), bins=15)
    ax.set_title(f"Histograma de {variable_hist}")
    ax.set_xlabel(variable_hist)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

if opcion == "Clasificación":
    col1, col2 = st.columns(2)

    if "class" in df.columns:
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x="class", data=df, ax=ax)
            ax.set_title("Distribución de la variable objetivo (class)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)

    variables_box = variables_numericas.copy()

    if "class" in df.columns and variables_box:
        with col2:
            variable_box = st.selectbox(
                "Selecciona una variable numérica para boxplot por clase",
                variables_box
            )
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(x="class", y=variable_box, data=df, ax=ax)
            ax.set_title(f"{variable_box} por clase")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)

else:
    col1, col2 = st.columns(2)

    if all(col in df.columns for col in ["age", "charges"]):
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.scatterplot(x="age", y="charges", data=df, ax=ax)
            ax.set_title("Edad vs cargos médicos")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)

    if all(col in df.columns for col in ["smoker", "charges"]):
        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(x="smoker", y="charges", data=df, ax=ax)
            ax.set_title("Cargos médicos según hábito de fumar")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)

# -------------------------
# Correlaciones
# -------------------------
st.subheader("Correlaciones")

if len(variables_numericas) >= 2:
    corr = df[variables_numericas].corr()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
else:
    st.write("No hay suficientes variables numéricas para calcular correlaciones.")

# -------------------------
# Imputación de datos faltantes
# -------------------------
st.subheader("Imputación de datos faltantes")

faltantes_antes = df.isnull().sum()
faltantes_antes_filtrados = faltantes_antes[faltantes_antes > 0]

st.write("Valores faltantes antes de la imputación:")
if len(faltantes_antes_filtrados) > 0:
    st.dataframe(faltantes_antes_filtrados)
else:
    st.info("No hay valores faltantes en esta base de datos.")

if opcion == "Regresión" and faltantes_antes.sum() == 0:
    st.success("La base de regresión no presenta valores faltantes, por lo tanto no requiere imputación.")
    df_imputado = df.copy()

elif opcion == "Clasificación" and faltantes_antes.sum() > 0:
    metodo = st.selectbox(
        "Selecciona el método de imputación",
        ["Simple", "KNN", "Iterativa"]
    )

    # Copias de trabajo para comparar todos los métodos
    data_original = df.copy()
    data_simple = df.copy()
    data_knn = df.copy()
    data_iter = df.copy()

    variables_numericas = df.select_dtypes(include="number").columns.tolist()
    variables_categoricas = df.select_dtypes(exclude="number").columns.tolist()

    # -------------------------
    # Imputación simple
    # -------------------------
    if variables_numericas:
        imp_num_simple = SimpleImputer(strategy="median")
        data_simple[variables_numericas] = imp_num_simple.fit_transform(data_simple[variables_numericas])

    if variables_categoricas:
        imp_cat_simple = SimpleImputer(strategy="most_frequent")
        data_simple[variables_categoricas] = imp_cat_simple.fit_transform(data_simple[variables_categoricas])

    # -------------------------
    # Imputación KNN
    # -------------------------
    if variables_categoricas:
        imp_cat_knn = SimpleImputer(strategy="most_frequent")
        data_knn[variables_categoricas] = imp_cat_knn.fit_transform(data_knn[variables_categoricas])

    if variables_numericas:
        imp_knn = KNNImputer(n_neighbors=5)
        data_knn[variables_numericas] = imp_knn.fit_transform(data_knn[variables_numericas])

    # -------------------------
    # Imputación iterativa
    # -------------------------
    if variables_categoricas:
        imp_cat_iter = SimpleImputer(strategy="most_frequent")
        data_iter[variables_categoricas] = imp_cat_iter.fit_transform(data_iter[variables_categoricas])

    if variables_numericas:
        imp_iter = IterativeImputer(random_state=42)
        data_iter[variables_numericas] = imp_iter.fit_transform(data_iter[variables_numericas])

    # -------------------------
    # Elegir cuál queda como df_imputado
    # -------------------------
    if metodo == "Simple":
        df_imputado = data_simple.copy()
    elif metodo == "KNN":
        df_imputado = data_knn.copy()
    else:
        df_imputado = data_iter.copy()

    faltantes_despues = df_imputado.isnull().sum()
    faltantes_despues_filtrados = faltantes_despues[faltantes_despues > 0]

    st.write(f"Método seleccionado: **{metodo}**")

    st.write("Valores faltantes después de la imputación:")
    if len(faltantes_despues_filtrados) > 0:
        st.dataframe(faltantes_despues_filtrados)
    else:
        st.success("No quedaron valores faltantes después de la imputación.")

    st.write("Vista previa de los datos imputados:")
    st.dataframe(df_imputado.head())

    # -------------------------
    # Comparación visual entre métodos
    # -------------------------
    st.markdown("### Comparación gráfica de métodos de imputación")

    variables_importantes = [col for col in ["age", "bp", "bgr", "bu", "sc", "hemo"] if col in variables_numericas]

    variable_comp = st.selectbox(
        "Selecciona una variable para comparar métodos",
        variables_importantes
    )

    col1, col2 = st.columns(2)

    # Gráfico de densidad
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.kdeplot(data_original[variable_comp].dropna(), label="Original", fill=True, ax=ax)
        sns.kdeplot(data_simple[variable_comp], label="Simple", fill=True, ax=ax)
        sns.kdeplot(data_knn[variable_comp], label="KNN", fill=True, ax=ax)
        sns.kdeplot(data_iter[variable_comp], label="Iterativa", fill=True, ax=ax)
        ax.set_title(f"Distribución comparativa de {variable_comp}")
        ax.set_xlabel(variable_comp)
        ax.set_ylabel("Densidad")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # Boxplot comparativo
    df_long = pd.concat([
        pd.DataFrame({"Metodo": "Original", "Valor": data_original[variable_comp].dropna()}),
        pd.DataFrame({"Metodo": "Simple", "Valor": data_simple[variable_comp]}),
        pd.DataFrame({"Metodo": "KNN", "Valor": data_knn[variable_comp]}),
        pd.DataFrame({"Metodo": "Iterativa", "Valor": data_iter[variable_comp]})
    ])

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=df_long, x="Metodo", y="Valor", ax=ax)
        ax.set_title(f"Boxplot comparativo de {variable_comp}")
        ax.set_xlabel("Método")
        ax.set_ylabel(variable_comp)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)

    # Conclusión del mejor método
    st.markdown("### Método recomendado")
    st.success(
        "El método recomendado para la base de clasificación es **KNN**, "
        "porque conserva mejor la forma de la distribución original y mantiene "
        "de forma más coherente la dispersión y los valores atípicos."
    )

elif opcion == "Clasificación" and faltantes_antes.sum() == 0:
    st.success("La base de clasificación no presenta valores faltantes, por lo tanto no requiere imputación.")
    df_imputado = df.copy()

else:
    df_imputado = df.copy()

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

    # ANOVA: charges según smoker
if all(col in df_imputado.columns for col in ["charges", "smoker"]):
    grupo_no = df_imputado[df_imputado["smoker"] == "no"]["charges"].dropna()
    grupo_yes = df_imputado[df_imputado["smoker"] == "yes"]["charges"].dropna()

    if len(grupo_no) > 0 and len(grupo_yes) > 0:
        stat, p = f_oneway(grupo_no, grupo_yes)

        st.write("**ANOVA: charges según smoker**")
        st.write(f"Estadístico F: {stat:.4f}")
        st.write(f"p-valor: {p:.6e}")
        if p < 0.05:
            st.success("Existen diferencias significativas en charges entre fumadores y no fumadores.")
        else:
            st.info("No se encontraron diferencias significativas en charges entre fumadores y no fumadores.")