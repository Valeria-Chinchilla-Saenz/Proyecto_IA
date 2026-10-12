import pandas as pd
from tensorflow.keras.models import load_model
from src import config


class PredictorDemanda:

    _COLUMNAS_ESPERADAS = [
        'holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed', 'season_1', 'season_2', 'season_3',
        'season_4', 'hr_0', 'hr_1', 'hr_2', 'hr_3', 'hr_4', 'hr_5', 'hr_6', 'hr_7', 'hr_8', 'hr_9', 'hr_10', 'hr_11',
        'hr_12', 'hr_13', 'hr_14', 'hr_15', 'hr_16', 'hr_17', 'hr_18', 'hr_19', 'hr_20', 'hr_21', 'hr_22', 'hr_23',
        'weathersit_1', 'weathersit_2', 'weathersit_3', 'weathersit_4', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3',
        'weekday_4', 'weekday_5', 'weekday_6'
    ]

    def __init__(self, ruta_modelo=None):
        ruta_modelo = ruta_modelo or (config.MODELS / "modelo1.keras")
        self._modelo = load_model(ruta_modelo)

    def _construir_fila(self, solicitud):

        datos = solicitud.dict()
        fila = pd.DataFrame([datos])

        fila = pd.get_dummies(fila, columns=["season", "weathersit", "hr", "weekday"])

        for columna in self._COLUMNAS_ESPERADAS:
            if columna not in fila.columns:
                fila[columna] = 0

        fila = fila[self._COLUMNAS_ESPERADAS]
        return fila

    def predecir(self, solicitud):
        fila = self._construir_fila(solicitud)
        prediccion = self._modelo.predict(fila)
        return float(prediccion[0][0])


class PredictorNivelDemanda:

    _COLUMNAS_ESPERADAS = [
        'holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed', 'season_1', 'season_2', 'season_3',
        'season_4', 'hr_0', 'hr_1', 'hr_2', 'hr_3', 'hr_4', 'hr_5', 'hr_6', 'hr_7', 'hr_8', 'hr_9', 'hr_10', 'hr_11',
        'hr_12', 'hr_13', 'hr_14', 'hr_15', 'hr_16', 'hr_17', 'hr_18', 'hr_19', 'hr_20', 'hr_21', 'hr_22', 'hr_23',
        'weathersit_1', 'weathersit_2', 'weathersit_3', 'weathersit_4', 'weekday_0', 'weekday_1', 'weekday_2',
        'weekday_3',
        'weekday_4', 'weekday_5', 'weekday_6'
    ]

    _ETIQUETAS = ["Baja", "Media", "Alta", "Pico"]

    def __init__(self, ruta_modelo=None):
        ruta_modelo = ruta_modelo or (config.MODELS / "modelo2.keras")
        self._modelo = load_model(ruta_modelo)

    def _construir_fila(self, solicitud):
        datos = solicitud.dict()
        fila = pd.DataFrame([datos])
        fila = pd.get_dummies(fila, columns=["season", "weathersit", "hr", "weekday"])

        for columna in self._COLUMNAS_ESPERADAS:
            if columna not in fila.columns:
                fila[columna] = 0

        fila = fila[self._COLUMNAS_ESPERADAS]
        return fila

    def predecir(self, solicitud):
        fila = self._construir_fila(solicitud)
        probabilidades = self._modelo.predict(fila)  # ej. [[0.05, 0.60, 0.30, 0.05]]
        indice_predicho = probabilidades.argmax(axis=1)[0]  # índice de la probabilidad más alta
        return self._ETIQUETAS[indice_predicho]