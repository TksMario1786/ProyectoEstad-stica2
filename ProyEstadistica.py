import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Visualización de Distribuciones", layout="wide")

st.title("Visualización de Distribuciones")


st.sidebar.header("Carga de Datos")
data_source = st.sidebar.radio("Selecciona la fuente de datos:", ("Cargar CSV", "Generar Datos Sintéticos"))

if data_source == "Cargar CSV":
    uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Por favor, sube un archivo CSV para continuar.")
        st.stop()
else:

    num_rows = st.sidebar.slider("Número de filas", 10, 1000, 100)
    num_cols = st.sidebar.slider("Número de columnas", 1, 10, 3)
    df = pd.DataFrame(np.random.randn(num_rows, num_cols), columns=[f"Col_{i+1}" for i in range(num_cols)])


st.subheader("Datos")
st.write(df)


st.sidebar.header("Selección de Variables")
selected_columns = st.sidebar.multiselect("Selecciona las variables a visualizar:", df.columns)

if not selected_columns:
    st.warning("Por favor, selecciona al menos una variable.")
    st.stop()

st.subheader("Visualización de Distribuciones")

chart_type = st.selectbox("Selecciona el tipo de gráfico:", ["Histograma", "Gráfico de Densidad", "Boxplot"])

for col in selected_columns:
    st.write(f"**Distribución de {col}**")
    if chart_type == "Histograma":
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Gráfico de Densidad":
        fig, ax = plt.subplots()
        sns.kdeplot(df[col], ax=ax)
        st.pyplot(fig)
    elif chart_type == "Boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)