import streamlit as st
import pandas as pd


st.set_page_config(page_title="Métricas de los Modelos", page_icon="📊", layout="wide")

st.title("📊 Evaluación y Comparación de Modelos ANN")
st.markdown("""
En esta sección se detallan las métricas obtenidas durante el entrenamiento y evaluación de los modelos.
Se comparan dos configuraciones para justificar la elección del mejor modelo.
""")

tab1, tab2 = st.tabs(
    ["Modelo 1: Regresión (Cantidad de Alquileres)", "Modelo 2: Clasificación Multiclase (Nivel de Demanda)"])

with tab1:
    st.header("Modelo 1 - Regresión (Variable Objetivo: cnt)")
    st.write("Comparación de métricas de error (MAE / RMSE) en el conjunto de prueba.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Configuración A (Con Dropout al 50% y dos capas)")
        st.metric(label="Error Absoluto Medio (MAE)", value="54.50 bicis")
        st.metric(label="Raíz del Error Cuadrático Medio (RMSE)", value="79.9 bicis")
    with col2:
        st.subheader("Configuración B (Con Dropout al 10% y tres capas)")
        st.metric(label="Error Absoluto Medio (MAE)", value="47.42 bicis", delta="-7.08", delta_color="inverse")
        st.metric(label="Raíz del Error Cuadrático Medio (RMSE)", value="70.32 bicis", delta="-9.58",
                  delta_color="inverse")

    st.info(
        "💡 **Justificación:** Se seleccionó la Configuración B ya que el uso de capas Dropout menor y el Early Stopping "
        "ayudó a prevenir el sobreajuste (overfitting), y la agregacion de una capa adicional a subir la precisión del modelo")

    st.markdown("### Curvas de Pérdida (Loss) vs Épocas")
    st.image("app/images/grafica_loss_regresion.png", caption="Curvas de Pérdida vs Épocas")

    st.markdown("### Valores reales vs Prediciones")
    st.image("app/images/valorreal_prediccion.png", caption="Valores reales vs Prediciones")

    st.markdown("### Acumulación errores")
    st.image("app/images/errores.png", caption="Acumulación errores")

    st.markdown("### Errores vs Predicciones")
    st.image("app/images/errores_predicciones.png", caption="Errores vs Predicciones")

with tab2:
    st.header("Modelo 2 - Clasificación (Nivel: Baja/Media/Alta/Pico)")
    st.write("Reporte de clasificación y matriz de confusión.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Configuración A (Arquitectura Simple)")
        st.metric(label="Precisión Global (Accuracy)", value="77%")
    with col4:
        st.subheader("Configuración B (Arquitectura Profunda)")
        st.metric(label="Precisión Global (Accuracy)", value="76%", delta="1%", delta_color="normal")

    st.markdown("### Reporte de Clasificación (Mejor Modelo)")
    # Datos de ejemplo, cámbialos por los que te dio tu classification_report
    reporte = pd.DataFrame({
        "Precisión (Precision)": [0.89, 0.76, 0.63, 0.80],
        "Exhaustividad (Recall)": [0.92, 0.69, 0.67, 0.79],
        "F1-Score": [0.90, 0.73, 0.65, 0.79]
    }, index=["Baja", "Media", "Alta", "Pico"])
    st.dataframe(reporte, use_container_width=True)

    st.markdown("### Matriz de Confusión")
    st.image("app/images/matriz_confusion.png", caption="Matriz de confusión - Conjunto de Prueba")