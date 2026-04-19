import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai
from io import StringIO

# Configuración inicial
st.set_page_config(page_title="Analizador Estadístico", layout="wide")
st.title("📊 Analizador Estadístico Interactivo")

# Configuración de Gemini (API Key)
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    modelo_ia = genai.GenerativeModel('gemini-pro')
    ia_disponible = True
except:
    ia_disponible = False
    st.warning("⚠️ La API de Gemini no está configurada. El módulo de IA no estará disponible.")

# =============================================
# MÓDULO 1: CARGA DE DATOS
# =============================================
st.sidebar.header("🔽 Carga de Datos")
fuente_datos = st.sidebar.radio(
    "Seleccione fuente de datos:",
    ("Subir archivo CSV", "Generar datos sintéticos"),
    index=0
)

datos = None

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
        except Exception as e:
            st.error(f"Error al leer el archivo: {str(e)}")
            st.stop()
    else:
        st.info("ℹ️ Por favor suba un archivo CSV para continuar")
        st.stop()
else:
    # Generación de datos sintéticos
    st.sidebar.subheader("Parámetros de Datos Sintéticos")
    num_filas = st.sidebar.slider("Número de filas", 30, 1000, 100)
    num_columnas = st.sidebar.slider("Número de columnas", 1, 5, 2)
    media = st.sidebar.number_input("Media", value=0.0)
    desviacion = st.sidebar.number_input("Desviación estándar", value=1.0, min_value=0.1)

    # Generar datos normales
    datos = pd.DataFrame(
        np.random.normal(loc=media, scale=desviacion, size=(num_filas, num_columnas)),
        columns=[f"Variable_{i + 1}" for i in range(num_columnas)]
    )
    st.success("✅ Datos sintéticos generados correctamente")

# Mostrar vista previa de datos
st.subheader("📋 Vista previa de los datos")
st.dataframe(datos.head(), height=200)

# =============================================
# MÓDULO 2: SELECCIÓN DE VARIABLES
# =============================================
st.sidebar.header("🎯 Selección de Variables")

if datos is not None:
    variables = datos.select_dtypes(include=['number']).columns.tolist()

    if not variables:
        st.error("❌ No se encontraron variables numéricas en los datos")
        st.stop()

    variable_analisis = st.sidebar.selectbox(
        "Seleccione variable para análisis:",
        variables,
        help="Seleccione la variable numérica que desea analizar"
    )

    # Extraer datos de la variable seleccionada
    datos_variable = datos[variable_analisis].dropna()
    n = len(datos_variable)

    if n < 30:
        st.warning(f"⚠️ Tamaño de muestra pequeño (n={n}). La prueba Z requiere n ≥ 30 para resultados confiables.")
else:
    st.stop()

# =============================================
# MÓDULO 3: VISUALIZACIÓN DE DISTRIBUCIONES
# =============================================
st.subheader("📈 Análisis Exploratorio de Datos")

# Crear pestañas para diferentes visualizaciones
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

# Análisis descriptivo
st.subheader("🔍 Resumen Estadístico")

# Calcular estadísticas
media_muestral = datos_variable.mean()
mediana = datos_variable.median()
desviacion_std = datos_variable.std()
sesgo = datos_variable.skew()
curtosis = datos_variable.kurtosis()
q1 = datos_variable.quantile(0.25)
q3 = datos_variable.quantile(0.75)
iqr = q3 - q1
lim_inf = q1 - 1.5 * iqr
lim_sup = q3 + 1.5 * iqr
outliers = datos_variable[(datos_variable < lim_inf) | (datos_variable > lim_sup)]

# Mostrar estadísticas en columnas
col1, col2 = st.columns(2)

with col1:
    st.metric("Media muestral", f"{media_muestral:.4f}")
    st.metric("Mediana", f"{mediana:.4f}")
    st.metric("Desviación estándar", f"{desviacion_std:.4f}")

with col2:
    st.metric("Sesgo", f"{sesgo:.4f}",
              help="Sesgo > 0: cola a la derecha, Sesgo < 0: cola a la izquierda")
    st.metric("Curtosis", f"{curtosis:.4f}",
              help="Curtosis > 0: más picuda que la normal, Curtosis < 0: más plana que la normal")
    st.metric("Número de outliers", len(outliers))

# Evaluación de normalidad
st.subheader("📏 Evaluación de Normalidad")

# Prueba de Shapiro-Wilk (para muestras pequeñas) o Kolmogorov-Smirnov (para muestras grandes)
if n < 50:
    _, p_normalidad = stats.shapiro(datos_variable)
    prueba_usada = "Shapiro-Wilk"
else:
    _, p_normalidad = stats.kstest(datos_variable, 'norm',
                                   args=(media_muestral, desviacion_std))
    prueba_usada = "Kolmogorov-Smirnov"

# Interpretación
normalidad = "SÍ" if p_normalidad > 0.05 else "NO"

st.write(f"**Prueba de {prueba_usada} para normalidad:**")
st.write(f"- Valor p: {p_normalidad:.4f}")
st.write(f"- ¿Distribución normal? **{normalidad}** (usando α=0.05)")

# Interpretación visual
st.write("**Interpretación visual:**")
if abs(sesgo) < 0.5 and abs(curtosis) < 1:
    st.success("✅ La distribución parece aproximadamente normal según el sesgo y curtosis")
elif abs(sesgo) >= 0.5:
    st.warning(f"⚠️ La distribución muestra sesgo ({'derecha' if sesgo > 0 else 'izquierda'})")
else:
    st.warning("⚠️ La distribución muestra curtosis significativa (muy picuda o muy plana)")

# =============================================
# MÓDULO 4: PRUEBA DE HIPÓTESIS Z
# =============================================
st.subheader("🧪 Prueba de Hipótesis Z")

# Configuración de la prueba
st.sidebar.header("⚙️ Parámetros de la Prueba Z")

# Inputs del usuario
media_hipotetica = st.sidebar.number_input(
    "Media hipotética (μ₀):",
    value=round(float(media_muestral), 2),
    step=0.01
)

sigma_conocida = st.sidebar.number_input(
    "Desviación estándar poblacional conocida (σ):",
    value=round(float(desviacion_std), 2),
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
error_std = sigma_conocida / np.sqrt(n)
z_estadistico = (media_muestral - media_hipotetica) / error_std

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

# Decisión
rechazar_H0 = p_valor < alpha

# Mostrar resultados
st.write("**Resultados de la Prueba Z:**")
col_res1, col_res2 = st.columns(2)

with col_res1:
    st.metric("Estadístico Z", f"{z_estadistico:.4f}")
    st.metric("Valor p", f"{p_valor:.4f}")

with col_res2:
    st.metric("Valor(es) crítico(s)", f"±{valor_critico:.4f}" if tipo_prueba == "Bilateral" else f"{valor_critico:.4f}")
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

# Gráfico de la distribución y región de rechazo
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

# =============================================
# MÓDULO 5: ASISTENTE DE IA (GEMINI)
# =============================================
if ia_disponible:
    st.subheader("🤖 Asistente de IA para Análisis Estadístico")

    # Crear resumen para la IA
    resumen_estadistico = f"""
    Resumen estadístico para análisis:
    - Variable analizada: {variable_analisis}
    - Tamaño de muestra (n): {n}
    - Media muestral: {media_muestral:.4f}
    - Media hipotética (μ₀): {media_hipotetica}
    - Desviación estándar poblacional (σ): {sigma_conocida}
    - Prueba de normalidad ({prueba_usada}): p-valor = {p_normalidad:.4f}
    - Sesgo: {sesgo:.4f}
    - Curtosis: {curtosis:.4f}
    - Número de outliers: {len(outliers)}

    Prueba Z realizada:
    - Hipótesis nula (H₀): μ = {media_hipotetica}
    - Hipótesis alternativa (H₁): {hipotesis}
    - Tipo de prueba: {tipo_prueba}
    - Nivel de significancia (α): {alpha}
    - Estadístico Z: {z_estadistico:.4f}
    - Valor p: {p_valor:.4f}
    - Decisión: {'Rechazar H₀' if rechazar_H0 else 'No rechazar H₀'}
    """

    # Prompt para la IA
    prompt_ia = f"""
    Eres un experto en estadística. Analiza los siguientes resultados y proporciona una interpretación clara y concisa:

    {resumen_estadistico}

    Por favor:
    1. Evalúa si los supuestos de la prueba Z son razonables
    2. Interpreta los resultados de la prueba de hipótesis
    3. Explica la decisión estadística en términos no técnicos
    4. Proporciona recomendaciones sobre posibles pasos siguientes

    Usa un lenguaje claro y evita jerga técnica excesiva. Responde en español.
    """

    if st.button("Consultar al Asistente de IA"):
        with st.spinner("Analizando resultados con IA..."):
            try:
                respuesta = modelo_ia.generate_content(prompt_ia)
                st.write("**Interpretación del Asistente de IA:**")
                st.write(respuesta.text)

                # Guardar consulta y respuesta
                st.session_state.ultima_respuesta_ia = respuesta.text
            except Exception as e:
                st.error(f"Error al consultar la IA: {str(e)}")

    if 'ultima_respuesta_ia' in st.session_state:
        with st.expander("Ver última respuesta de IA"):
            st.write(st.session_state.ultima_respuesta_ia)
else:
    st.warning("El módulo de IA no está disponible. Configura la API de Gemini para habilitarlo.")

# =============================================
# INFORMACIÓN ADICIONAL
# =============================================
st.sidebar.header("ℹ️ Acerca de")
st.sidebar.info("""
Esta aplicación permite:
- Visualizar distribuciones de datos
- Realizar pruebas de hipótesis Z
- Consultar un asistente de IA para interpretación

Desarrollado para fines educativos.
""")