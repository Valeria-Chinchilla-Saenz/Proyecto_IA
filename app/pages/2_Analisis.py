import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Análisis de Lotes", page_icon="📈", layout="wide")

st.title("📈 Análisis de Lotes y Proyección Horaria")
st.markdown("""
Sube un archivo CSV con un lote de datos (por ejemplo, el clima y calendario de las próximas 24 horas). 
El sistema realizará predicciones masivas para proyectar la demanda por franja horaria y sugerir la redistribución de la flota.
""")

# 1. Carga de Archivo (Lote)
archivo_subido = st.file_uploader("Sube tu archivo de proyecciones (CSV)", type=["csv"])

# URL de tu API
API_URL_BASE = "http://localhost:8000"

if archivo_subido is not None:
    # Leer el CSV
    df = pd.read_csv(archivo_subido)
    st.write("### Vista previa del lote cargado:")
    st.dataframe(df.head())

    # Validar que tenga la columna 'hr' (hora) para la gráfica
    if 'hr' not in df.columns:
        st.error("El archivo CSV debe contener una columna llamada 'hr' (hora) para graficar la franja horaria.")
    else:
        if st.button("🚀 Ejecutar Predicción de Lote"):
            with st.spinner("Procesando lote de datos a través de la API..."):

                # Listas para guardar los resultados
                predicciones_cnt = []
                predicciones_nivel = []

                # Iterar sobre las filas del dataframe para hacer la predicción
                # Nota: Si programaste un endpoint /predict/batch en tu FastAPI,
                # es mejor enviar el JSON completo de una vez. Aquí iteramos asumiendo endpoints individuales.
                for index, row in df.iterrows():
                    datos_entrada = {
                        "season": int(row.get("season", 1)),
                        "mnth": int(row.get("mnth", 1)),
                        "hr": int(row.get("hr", 0)),
                        "holiday": int(row.get("holiday", 0)),
                        "weekday": int(row.get("weekday", 0)),
                        "workingday": int(row.get("workingday", 1)),
                        "weathersit": int(row.get("weathersit", 1)),
                        "temp": float(row.get("temp", 0.5)),
                        "atemp": float(row.get("atemp", 0.5)),
                        "hum": float(row.get("hum", 0.5)),
                        "windspeed": float(row.get("windspeed", 0.2))
                    }

                    # Llamada al modelo de Regresión
                    try:
                        res_reg = requests.post(f"{API_URL_BASE}/predict/demand", json=datos_entrada)
                        if res_reg.status_code == 200:
                            predicciones_cnt.append(res_reg.json().get('demanda_estimada', 0))
                        else:
                            predicciones_cnt.append(0)

                        # Llamada al modelo de Clasificación
                        res_clas = requests.post(f"{API_URL_BASE}/predict/demand_level", json=datos_entrada)
                        if res_clas.status_code == 200:
                            predicciones_nivel.append(res_clas.json().get('nivel_demanda', 'Desconocido'))
                        else:
                            predicciones_nivel.append("Error")

                    except Exception as e:
                        st.error("Error de conexión con la API.")
                        break

                # Añadir los resultados al DataFrame
                if len(predicciones_cnt) == len(df):
                    df['Demanda_Proyectada'] = predicciones_cnt
                    df['Nivel_Alerta'] = predicciones_nivel

                    st.success("✅ ¡Predicción masiva completada!")

                    # Mostrar resultados tabulares
                    st.write("### Resultados del Lote")
                    st.dataframe(df[['hr', 'temp', 'weathersit', 'Demanda_Proyectada', 'Nivel_Alerta']])

                    # 2. Proyección de demanda por franja horaria (Requisito del PDF)
                    st.header("📉 Proyección de Demanda por Franja Horaria")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    sns.lineplot(data=df, x='hr', y='Demanda_Proyectada', marker="o", color="blue", ax=ax)
                    ax.set_title("Proyección de Alquileres a lo largo del día")
                    ax.set_xlabel("Hora del Día")
                    ax.set_ylabel("Cantidad de Bicicletas Estimadas")
                    ax.set_xticks(range(0, 24))
                    ax.grid(True, linestyle="--", alpha=0.6)
                    st.pyplot(fig)

                    # 3. Recomendaciones de redistribución de la flota (Requisito del PDF)
                    st.header("🚚 Recomendaciones de Redistribución de Flota")

                    horas_pico = df[df['Nivel_Alerta'] == 'Pico']['hr'].tolist()
                    horas_alta = df[df['Nivel_Alerta'] == 'Alta']['hr'].tolist()

                    if horas_pico or horas_alta:
                        st.warning("⚠️ **Alerta de Alta Demanda Detectada**")
                        if horas_pico:
                            st.write(
                                f"- **Urgente:** Se espera nivel **PICO** a las horas: {horas_pico}. Se recomienda enviar camiones de redistribución 1 hora antes a las estaciones principales (centros de trabajo/estudio).")
                        if horas_alta:
                            st.write(
                                f"- **Atención:** Se espera nivel **ALTO** a las horas: {horas_alta}. Monitorear disponibilidad de anclajes libres.")
                    else:
                        st.info(
                            "✅ La demanda proyectada es manejable con la distribución actual de la flota. No se requieren traslados de emergencia.")