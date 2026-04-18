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

