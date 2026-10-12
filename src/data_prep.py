import pandas as pd

class CargadorDatos:

    def __init__(self, ruta_train, ruta_test, columna_objetivo="cnt", columnas_excluir=None):
        self._ruta_train = ruta_train
        self._ruta_test = ruta_test
        self._columna_objetivo = columna_objetivo
        self._columnas_excluir = columnas_excluir or []
        self._train = None
        self._test = None

    def _cargar_csv(self):
        self._train = pd.read_csv(self._ruta_train)
        self._test = pd.read_csv(self._ruta_test)

    def obtener_datos(self):
        self._cargar_csv()

        for nombre, datos in [("train", self._train), ("test", self._test)]:
            if self._columna_objetivo not in datos.columns:
                raise ValueError(f"La columna '{self._columna_objetivo}' no existe en {nombre}.")

        columnas_quitar = list(dict.fromkeys([self._columna_objetivo] + self._columnas_excluir))

        X_train = self._train.drop(columns=columnas_quitar, errors="ignore")
        y_train = self._train[[self._columna_objetivo]]
        X_test = self._test.drop(columns=columnas_quitar, errors="ignore")
        y_test = self._test[[self._columna_objetivo]]

        return X_train, y_train, X_test, y_test

