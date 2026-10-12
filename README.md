# Proyecto_IA
# Sistema de Predicción de Demanda de Bicicletas (Proyecto E)

Este proyecto utiliza Redes Neuronales Artificiales (ANN) para predecir la demanda de alquiler de bicicletas compartidas basándose en condiciones de calendario y clima.

## 👥 Equipo de Trabajo
* **Persona A:** Datos, Preprocesamiento, Frontend de Análisis y Documentación.
* **Persona B:** Modelo de Regresión y Endpoint API de cantidad.
* **Persona C:** Modelo de Clasificación y Endpoint API de niveles.

##  Guía de Instalación y Uso
1. Clonar el repositorio.
2. Instalar las dependencias usando `%pip install -r ../requirements.txt`.
3. Para ejecutar la API: `uvicorn api.main:app --reload`
4. Para ejecutar el Dashboard: `streamlit run app/Home.py`

## 📁 Estructura del Proyecto
* `/data`: Datasets originales y procesados.
* `/models`: Modelos ANN entrenados y el MinMaxScaler (`scaler.pkl`).
* `/api`: Código de la API REST con FastAPI.
* `/app`: Frontend web multipágina construido con Streamlit.

## Dataset 
* Fuente: Bike Sharing Dataset (UCI Machine Learning Repository)
* URL: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
* Registros: 17,379
* Variables: 17 (calendario, clima y conteos climaticos)

## 📊 Resultados de los Modelos
Modelo 1 - Regresión:

  Objetivo: Predecir la cantidad de alquileres por hora
  
  Arquitectura: Red neuronal con dos capas ocultas y Dropout al 50%
  
  Métricas:
  
  * MAE: 47.42
  
  * RSE: 70.32

   * Varianza Explicada: 0.8439
    
  * R2: 0.8439
  
  Modelo 2 - Clasificación Multiclase:
  
  Objetivo: Clasificar nivel de demanda en cuatro categorías (Baja, Media, Alta, Pico)
  
  Arquitectura: Red neuronal simple (contiene dos capas).
  
  Métricas:
  
  * Accuracy: 0.76

  * Precisión promedio: 0.77
  
  * Recall promedio: 0.77
  
  * F1-score promedio: 0.77

## 📈 Comparación de los resultados

El modelo de regresión obtiene una mayor capacidad (R² ≈ 0.80) al trabajar con una variable, en cambio el modelo multiclase enfrenta más compleja al segmentar la demanda de bicicletas en cuatro niveles. A pesar de que el modelo regresión logra mayor precisión numérica, el modelo de clasificación aporta un valor analítico superior al permitir categorizar patrones de comportamiento y facilitar decisiones estratégicas en contextos de negocio.

## 💡 Conclusiones

Como se pudo observar en el EDA se reveló que la demanda de bicicletas está fuertemente influenciada por la hora del día, registrando alta demanda a las 8:00 AM, y entre las 5:00 PM y 6:00 PM, así como por factores climáticos donde los días despejados favorecen el uso y las lluvias o tormentas lo reducen drásticamente. Para manejar estos cambios en los datos, implementamos dos modelos de Redes Neuronales ajustados cada uno en sus capas para evitar sobreajuste: el modelo de regresión, predice la cantidad de bicicletas por hora, y el modelo de clasificación multiclase, categoriza el nivel de demanda (Baja/Media/Alta/Pico).

## Recomendaciones
El modelo de regresión muestra una limitación específica prediciendo demanda alta y pico, por la menor cantidad de datos de entrenamiento en ese rango, por lo que mas datos de este rango hubieran mejorado los resultados del modelo. También el ajuste del learning rate, podría ayudar mejorando los resultados de ambos modelos al bajar los saltos o como cubre este cada dato pero al final no se implemento. Además al hipotéticamente existir mas variables climáticas y temporales la precisión del mmodelo podría mejorar.
         
