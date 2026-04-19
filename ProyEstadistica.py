import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests  # Importamos requests para hacer llamadas HTTP

# Configuración inicial
st.set_page_config(page_title="Análisis Estadístico con Gemini", layout="wide")

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
api_key = st.secrets["GEMINI_API_KEY"]


def generar_respuesta_ia(prompt):
    headers = {
        "Content-Type": "application/json",
    }
    params = {
        "key": api_key
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, params=params, json=payload)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"🚨 Error en la API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"🚨 Error al conectarse a la API: {str(e)}"


def asistente_ia(stats_dict, prueba_dict, variable_analisis):
    st.subheader("🤖 Asistente de IA para Análisis Estadístico")

    resumen_estadistico = f"""
              Resumen estadístico para análisis:
              - Variable: {variable_analisis}
              - n: {stats_dict['n']}
              - Media muestral: {stats_dict['media']:.4f}
              - Media hipotética (μ₀): {prueba_dict['media_hipotetica']}
              - P-valor normalidad: {stats_dict['p_normalidad']:.4f}
              - Estadístico Z: {prueba_dict['z_estadistico']:.4f}
              - Valor p prueba: {prueba_dict['p_valor']:.4f}
              - Decisión: {'Rechazar H₀' if prueba_dict['p_valor'] < prueba_dict['alpha'] else 'No rechazar H₀'}
              """

    if st.button("Consultar al Asistente de IA"):
        with st.spinner("Miku está analizando los datos..."):
            prompt_completo = f"""
            Eres Miku Nakano. Eres tímida pero experta en datos. 
            Analiza estos resultados: {resumen_estadistico}
            1. Evalúa supuestos. 2. Interpreta resultados. 3. Da una conclusión clara.
            Responde en español con tu personalidad característica.
            """

            respuesta = generar_respuesta_ia(prompt_completo)
            st.session_state.ultima_respuesta_ia = respuesta
            st.write(respuesta)

    if 'ultima_respuesta_ia' in st.session_state:
        with st.expander("Ver última respuesta"):
            st.write(st.session_state.ultima_respuesta_ia)


# Módulo 1: Carga de datos
def cargar_datos():
    st.sidebar.header("🔽 Carga de Datos")
    fuente_datos = st.sidebar.radio(
        "Seleccione fuente de datos:",
        ("Subir archivo CSV", "Generar datos sintéticos"),
        index=0
    )

    if fuente_datos == "Subir archivo CSV":
        archivo = st.sidebar.file_uploader(
            "Suba su archivo CSV",
            type=["csv"],
            help="El archivo debe contener datos numéricos para el análisis"
        )

        if archivo is not None:
            try:
                datos = pd.read_csv(archivo)
                st.success("✅ Archivo cargado correctamente")
                return datos
            except Exception as e:
                st.error(f"Error al leer el archivo: {str(e)}")
                st.stop()
        else:
            st.info("ℹ️ Por favor suba un archivo CSV para continuar")
            st.stop()
    else:
        return generar_datos_sinteticos()


def generar_datos_sinteticos():
    st.sidebar.subheader("Parámetros de Datos Sintéticos")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        num_filas = st.number_input("Número de filas", min_value=1, value=100, step=1)
    with col2:
        num_columnas = st.number_input("Número de columnas", min_value=1, max_value=10, value=2, step=1)

    media = st.sidebar.number_input("Media", value=0.0)
    desviacion = st.sidebar.number_input("Desviación estándar", value=1.0, min_value=0.1)

    datos = pd.DataFrame(
        np.random.normal(loc=media, scale=desviacion, size=(num_filas, num_columnas)),
        columns=[f"Variable_{i + 1}" for i in range(num_columnas)]
    )
    st.success("✅ Datos sintéticos generados correctamente")
    return datos


# Módulo 2: Selección de variables
def seleccionar_variable(datos):
    st.sidebar.header("🎯 Selección de Variables")
    variables = datos.select_dtypes(include=['number']).columns.tolist()

    if not variables:
        st.error("❌ No se encontraron variables numéricas en los datos")
        st.stop()

    variable_analisis = st.sidebar.selectbox(
        "Seleccione variable para análisis:",
        variables,
        help="Seleccione la variable numérica que desea analizar"
    )

    datos_variable = datos[variable_analisis].dropna()
    n = len(datos_variable)

    if n < 30:
        st.warning(f"⚠️ Tamaño de muestra pequeño (n={n}). La prueba Z requiere n ≥ 30 para resultados confiables.")

    return datos_variable, n, variable_analisis


# Módulo 3: Visualización de distribuciones
def visualizar_distribuciones(datos_variable, variable_analisis):
    st.subheader("📈 Análisis Exploratorio de Datos")
    tab1, tab2, tab3 = st.tabs(["Histograma", "Gráfico de Densidad (KDE)", "Diagrama de Caja"])

    with tab1:
        st.write(f"**Histograma de {variable_analisis}**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(datos_variable, kde=False, bins='auto', ax=ax)
        ax.set_xlabel(variable_analisis)
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    with tab2:
        st.write(f"**Gráfico de Densidad (KDE) de {variable_analisis}**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(datos_variable, fill=True, ax=ax)
        ax.set_xlabel(variable_analisis)
        ax.set_ylabel("Densidad")
        st.pyplot(fig)

    with tab3:
        st.write(f"**Diagrama de Caja de {variable_analisis}**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=datos_variable, ax=ax)
        ax.set_xlabel(variable_analisis)
        st.pyplot(fig)


# Módulo 4: Análisis descriptivo y normalidad
def analisis_descriptivo(datos_variable, variable_analisis, n):
    st.subheader("🔍 Resumen Estadístico")

    # Calcular estadísticas
    stats_dict = {
        'media': datos_variable.mean(),
        'mediana': datos_variable.median(),
        'desviacion_std': datos_variable.std(),
        'sesgo': datos_variable.skew(),
        'curtosis': datos_variable.kurtosis(),
        'q1': datos_variable.quantile(0.25),
        'q3': datos_variable.quantile(0.75),
        'n': n
    }

    # Calcular outliers
    iqr = stats_dict['q3'] - stats_dict['q1']
    lim_inf = stats_dict['q1'] - 1.5 * iqr
    lim_sup = stats_dict['q3'] + 1.5 * iqr
    outliers = datos_variable[(datos_variable < lim_inf) | (datos_variable > lim_sup)]
    stats_dict['outliers'] = len(outliers)

    # Mostrar estadísticas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Media muestral", f"{stats_dict['media']:.4f}")
        st.metric("Mediana", f"{stats_dict['mediana']:.4f}")
        st.metric("Desviación estándar", f"{stats_dict['desviacion_std']:.4f}")

    with col2:
        st.metric("Sesgo", f"{stats_dict['sesgo']:.4f}",
                  help="Sesgo > 0: cola a la derecha, Sesgo < 0: cola a la izquierda")
        st.metric("Curtosis", f"{stats_dict['curtosis']:.4f}",
                  help="Curtosis > 0: más picuda que la normal, Curtosis < 0: más plana que la normal")
        st.metric("Número de outliers", stats_dict['outliers'])

    # Evaluación de normalidad
    st.subheader("📏 Evaluación de Normalidad")
    if n < 50:
        _, p_normalidad = stats.shapiro(datos_variable)
        prueba_usada = "Shapiro-Wilk"
    else:
        _, p_normalidad = stats.kstest(datos_variable, 'norm',
                                       args=(stats_dict['media'], stats_dict['desviacion_std']))
        prueba_usada = "Kolmogorov-Smirnov"

    normalidad = "SÍ" if p_normalidad > 0.05 else "NO"
    stats_dict.update({'p_normalidad': p_normalidad, 'prueba_usada': prueba_usada, 'normalidad': normalidad})

    st.write(f"**Prueba de {prueba_usada} para normalidad:**")
    st.write(f"- Valor p: {p_normalidad:.4f}")
    st.write(f"- ¿Distribución normal? **{normalidad}** (usando α=0.05)")

    # Interpretación visual
    st.write("**Interpretación visual:**")
    if abs(stats_dict['sesgo']) < 0.5 and abs(stats_dict['curtosis']) < 1:
        st.success("✅ La distribución parece aproximadamente normal según el sesgo y curtosis")
    elif abs(stats_dict['sesgo']) >= 0.5:
        st.warning(f"⚠️ La distribución muestra sesgo ({'derecha' if stats_dict['sesgo'] > 0 else 'izquierda'})")
    else:
        st.warning("⚠️ La distribución muestra curtosis significativa (muy picuda o muy plana)")

    return stats_dict


# Módulo 5: Prueba de hipótesis Z
def prueba_hipotesis_z(stats_dict, variable_analisis):
    st.subheader("🧪 Prueba de Hipótesis Z")
    st.sidebar.header("⚙️ Parámetros de la Prueba Z")

    # Inputs del usuario
    media_hipotetica = st.sidebar.number_input(
        "Media hipotética (μ₀):",
        value=round(float(stats_dict['media']), 2),
        step=0.01
    )

    sigma_conocida = st.sidebar.number_input(
        "Desviación estándar poblacional conocida (σ):",
        value=round(float(stats_dict['desviacion_std']), 2),
        min_value=0.01,
        step=0.01,
        help="Si no se conoce, usar la desviación estándar muestral como estimación"
    )

    tipo_prueba = st.sidebar.radio(
        "Tipo de prueba:",
        ("Bilateral", "Cola izquierda", "Cola derecha"),
        index=0,
        help="Bilateral: μ ≠ μ₀, Cola izquierda: μ < μ₀, Cola derecha: μ > μ₀"
    )

    alpha = st.sidebar.slider(
        "Nivel de significancia (α):",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01
    )

    # Realizar prueba Z
    error_std = sigma_conocida / np.sqrt(stats_dict['n'])
    z_estadistico = (stats_dict['media'] - media_hipotetica) / error_std

    if tipo_prueba == "Bilateral":
        p_valor = 2 * (1 - stats.norm.cdf(abs(z_estadistico)))
        valor_critico = stats.norm.ppf(1 - alpha / 2)
        region_rechazo = f"Z < -{valor_critico:.4f} o Z > {valor_critico:.4f}"
    elif tipo_prueba == "Cola izquierda":
        p_valor = stats.norm.cdf(z_estadistico)
        valor_critico = stats.norm.ppf(alpha)
        region_rechazo = f"Z < {valor_critico:.4f}"
    else:  # Cola derecha
        p_valor = 1 - stats.norm.cdf(z_estadistico)
        valor_critico = stats.norm.ppf(1 - alpha)
        region_rechazo = f"Z > {valor_critico:.4f}"

    rechazar_H0 = p_valor < alpha

    # Mostrar resultados
    st.write("**Resultados de la Prueba Z:**")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric("Estadístico Z", f"{z_estadistico:.4f}")
        st.metric("Valor p", f"{p_valor:.4f}")

    with col_res2:
        st.metric("Valor(es) crítico(s)",
                  f"±{valor_critico:.4f}" if tipo_prueba == "Bilateral" else f"{valor_critico:.4f}")
        st.metric("Decisión",
                  "Rechazar H₀" if rechazar_H0 else "No rechazar H₀",
                  delta="Significativo" if rechazar_H0 else "No significativo",
                  delta_color="inverse")

    st.write(f"**Región de rechazo:** {region_rechazo}")

    # Interpretación
    st.write("**Interpretación:**")
    if tipo_prueba == "Bilateral":
        hipotesis = f"μ ≠ {media_hipotetica}"
    elif tipo_prueba == "Cola izquierda":
        hipotesis = f"μ < {media_hipotetica}"
    else:
        hipotesis = f"μ > {media_hipotetica}"

    if rechazar_H0:
        st.success(
            f"Con un nivel de significancia de α={alpha}, hay evidencia suficiente para rechazar H₀ a favor de H₁: {hipotesis}")
    else:
        st.info(f"Con un nivel de significancia de α={alpha}, no hay evidencia suficiente para rechazar H₀.")

    # Gráfico de la distribución
    st.subheader("📊 Visualización de la Prueba Z")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-4, 4, 1000) if tipo_prueba == "Bilateral" else np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)

    ax.plot(x, y, label='Distribución Normal Estándar')
    ax.axvline(z_estadistico, color='red', linestyle='--', label=f'Z observado = {z_estadistico:.2f}')

    if tipo_prueba == "Bilateral":
        ax.fill_between(x, y, where=(x < -valor_critico) | (x > valor_critico),
                        color='red', alpha=0.3, label='Región de Rechazo')
        ax.axvline(-valor_critico, color='green', linestyle=':', label=f'Z crítico = ±{valor_critico:.2f}')
        ax.axvline(valor_critico, color='green', linestyle=':')
    else:
        if tipo_prueba == "Cola izquierda":
            ax.fill_between(x, y, where=(x < valor_critico),
                            color='red', alpha=0.3, label='Región de Rechazo')
        else:
            ax.fill_between(x, y, where=(x > valor_critico),
                            color='red', alpha=0.3, label='Región de Rechazo')
        ax.axvline(valor_critico, color='green', linestyle=':', label=f'Z crítico = {valor_critico:.2f}')

    ax.set_title('Distribución del Estadístico Z bajo H₀')
    ax.set_xlabel('Valor Z')
    ax.set_ylabel('Densidad')
    ax.legend()
    st.pyplot(fig)

    return {
        'media_hipotetica': media_hipotetica,
        'sigma_conocida': sigma_conocida,
        'tipo_prueba': tipo_prueba,
        'alpha': alpha,
        'z_estadistico': z_estadistico,
        'p_valor': p_valor,
        'hipotesis': hipotesis
    }


# --- FUNCIÓN PRINCIPAL ---
def main():
    st.title("📊 Análisis Estadístico + Miku AI")

    datos = cargar_datos()
    if datos is not None:
        datos_var, n, var_nombre = seleccionar_variable(datos)

        s_dict = analisis_descriptivo(datos_var, var_nombre, n)
        p_dict = prueba_hipotesis_z(s_dict, var_nombre)

        st.write(f"Media: {s_dict['media']:.2f} | P-valor: {p_dict['p_valor']:.4f}")

        asistente_ia(s_dict, p_dict, var_nombre)
    else:
        st.info("Por favor, sube un archivo CSV en la barra lateral.")


if __name__ == "__main__":
    main()