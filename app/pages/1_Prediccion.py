import streamlit as st
import requests

st.set_page_config(page_title="Predicción de Demanda", page_icon="🔮", layout="wide")

st.title("🔮 Predicción de Demanda de Bicicletas")
st.markdown("Ingresa las condiciones del clima y calendario para predecir la demanda de alquileres.")

# Formulario agrupando variables
with st.form("form_prediccion"):
    st.subheader("📅 Datos de Calendario")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        season = st.selectbox("Temporada (season)", [1, 2, 3, 4],
                              format_func=lambda x: {1: "Invierno", 2: "Primavera", 3: "Verano", 4: "Otoño"}[x])
    with col2:
        mnth = st.slider("Mes (mnth)", 1, 12, 6)
    with col3:
        hr = st.slider("Hora (hr)", 0, 23, 12)
    with col4:
        weekday = st.slider("Día de la semana (weekday)", 0, 6, 3)

    col_a, col_b = st.columns(2)
    with col_a:
        holiday = st.selectbox("¿Es Feriado? (holiday)", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    with col_b:
        workingday = st.selectbox("¿Es Día Laboral? (workingday)", [0, 1],
                                  format_func=lambda x: "Sí" if x == 1 else "No")

    st.subheader("🌤️ Datos Climáticos")
    col5, col6 = st.columns(2)
    with col5:
        weathersit = st.selectbox("Situación Climática (weathersit)", [1, 2, 3, 4],
                                  format_func=lambda x: ["Despejado", "Nublado", "Lluvia Ligera", "Tormenta Fuerte"][
                                      x - 1])
        temp = st.slider("Temperatura normalizada (temp)", 0.0, 1.0, 0.5)
        atemp = st.slider("Sensación Térmica norm. (atemp)", 0.0, 1.0, 0.5)
    with col6:
        hum = st.slider("Humedad normalizada (hum)", 0.0, 1.0, 0.5)
        windspeed = st.slider("Velocidad del Viento norm. (windspeed)", 0.0, 1.0, 0.2)

    # Botones de envío
    submit_regresion = st.form_submit_button("🚲 Estimar Cantidad de Alquileres (Regresión)")
    submit_clasificacion = st.form_submit_button("📊 Predecir Nivel de Demanda (Clasificación)")

# Diccionario con los datos a enviar a la API
datos_entrada = {
    "season": season,
    "mnth": mnth,
    "hr": hr,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "weathersit": weathersit,
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed
}

# Lógica para llamar a la API
API_URL_BASE = "http://localhost:8000"

if submit_regresion:
    try:
        with st.spinner("Calculando predicción..."):
            respuesta = requests.post(f"{API_URL_BASE}/predict/demand", json=datos_entrada)
            if respuesta.status_code == 200:
                resultado = respuesta.json()
                # Ajusta la clave "cantidad_estimada" según lo que devuelva tu API en FastAPI
                st.success(
                    f"📈 **Cantidad estimada de alquileres:** {resultado.get('demanda_estimada', 'No disponible'):.0f} bicicletas")

            else:
                st.error(f"Error de la API: {respuesta.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar a la API. Asegúrate de que FastAPI esté en ejecución.")

if submit_clasificacion:
    try:
        with st.spinner("Calculando nivel de demanda..."):
            respuesta = requests.post(f"{API_URL_BASE}/predict/demand_level", json=datos_entrada)
            if respuesta.status_code == 200:
                resultado = respuesta.json()
                # Ajusta la clave según tu API (Baja/Media/Alta/Pico)
                nivel = resultado.get('nivel_demanda', 'No disponible')
                st.info(f"🚦 **Nivel de demanda proyectado:** {nivel}")

                # Recomendación de flota (ejemplo)
                if nivel == "Alta" or nivel == "Pico":
                    st.warning(
                        "⚠️ **Recomendación:** Se sugiere redistribuir la flota y agregar unidades extra a las estaciones.")
            else:
                st.error(f"Error de la API: {respuesta.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar a la API. Asegúrate de que FastAPI esté en ejecución.")