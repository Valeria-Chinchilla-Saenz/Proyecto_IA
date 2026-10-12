import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from src import config


class PredictorDemanda:

    def __init__(self, ruta_modelo=None, ruta_columnas=None):
        ruta_modelo = ruta_modelo or (config.MODELS / "modelo1.keras")
        ruta_columnas = ruta_columnas or (config.MODELS / "columnas_modelo1.pkl")

        self._modelo = load_model(ruta_modelo)
        with open(ruta_columnas, "rb") as archivo:
            self._columnas_esperadas = pickle.load(archivo)

    def _construir_fila(self, solicitud):
        datos = solicitud.dict()
        fila = pd.DataFrame([datos])
        fila = pd.get_dummies(fila, columns=["season", "weathersit", "hr", "weekday"])
        fila = fila.reindex(columns=self._columnas_esperadas, fill_value=0)
        return fila

    def predecir(self, solicitud):
        fila = self._construir_fila(solicitud)
        prediccion = self._modelo.predict(fila, verbose=0)
        return float(prediccion[0][0])


class PredictorNivelDemanda:

    _ETIQUETAS = ["Baja", "Media", "Alta", "Pico"]

    def __init__(self, ruta_modelo=None, ruta_columnas=None):
        ruta_modelo = ruta_modelo or (config.MODELS / "modelo2.keras")
        ruta_columnas = ruta_columnas or (config.MODELS / "columnas_modelo2.pkl")

        self._modelo = load_model(ruta_modelo)
        with open(ruta_columnas, "rb") as archivo:
            self._columnas_esperadas = pickle.load(archivo)

    def _construir_fila(self, solicitud):
        datos = solicitud.dict()
        fila = pd.DataFrame([datos])
        fila = pd.get_dummies(fila, columns=["season", "weathersit", "hr", "weekday"])
        fila = fila.reindex(columns=self._columnas_esperadas, fill_value=0)
        return fila

    def predecir(self, solicitud):
        fila = self._construir_fila(solicitud)
        probabilidades = self._modelo.predict(fila, verbose=0)
        indice_predicho = probabilidades.argmax(axis=1)[0]
        return self._ETIQUETAS[indice_predicho]