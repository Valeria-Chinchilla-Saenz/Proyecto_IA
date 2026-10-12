import streamlit as st

st.set_page_config(
    page_title="Demanda de Bicicletas | IA Aplicada",
    page_icon="🚲",
    layout="wide"
)

# Encabezado principal
st.title("🚲 Sistema Inteligente de Predicción de Demanda de Bicicletas")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### Sobre el Proyecto
        Este sistema utiliza **Redes Neuronales Artificiales (ANN)** para predecir 
        la demanda de alquiler de bicicletas compartidas basándose en condiciones 
        de calendario y clima. 

        El objetivo principal es estimar la cantidad de alquileres por hora y clasificar 
        el nivel de demanda para ayudar a planificar la **distribución óptima de la flota**.
        """
    )

with col2:

    st.info(
        """
        **🎓 Información Académica**
        * **Integrantes:** Brandon Abarca / Valeria Chinchilla / Roberto Gonzalez
        * **Institución:** Colegio Universitario de Cartago (CUC)
        * **Carrera:** Big Data (BD)
        * **Curso:** BD-151 Inteligencia Artificial Aplicada
        * **Profesor:** Osvaldo Gonzalez Chaves
        * **Año:** 2026
        """
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧭 Guía de Navegación")
st.write("Utiliza el menú lateral izquierdo para explorar los diferentes módulos del sistema:")


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.success(
        "🔮 **1. Predicción**\n\nRealiza estimaciones manuales ingresando condiciones climáticas y de calendario.")
with c2:
    st.info(
        "📈 **2. Análisis (Lotes)**\n\nSube un archivo CSV para generar proyecciones por franja horaria y ver alertas de flota.")
with c3:
    st.warning(
        "📊 **3. Métricas**\n\nCompara el rendimiento de los modelos ANN, evalúa el MAE, RMSE y las matrices de confusión.")
with c4:
    st.error(
        "📉 **4. EDA**\n\nExplora los patrones históricos de demanda y la influencia del clima mediante visualizaciones.")

st.markdown("---")
st.caption("Desarrollado para el Proyecto E: Sistema de Predicción de Demanda de Bicicletas.")