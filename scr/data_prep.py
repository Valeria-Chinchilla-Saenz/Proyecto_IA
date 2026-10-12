import pandas as pd

class CargadorDatosModelo1:

    def __init__(self, ruta_train, ruta_test, columna_objetivo="cnt"):
        self._ruta_train = ruta_train
        self._ruta_test = ruta_test
        self._columna_objetivo = columna_objetivo
        self._train = None
        self._test = None

    def _cargar_csv(self):
        self._train = pd.read_csv(self._ruta_train)
        self._test = pd.read_csv(self._ruta_test)

    def _codificar_weekday(self):
        # Se codifica train y test JUNTOS para evitar el problema que ya
        # vimos antes: si se codifican por separado, puede faltar alguna
        # columna en uno de los dos si no aparece esa categoria ahi.
        n_train = len(self._train)
        combinado = pd.concat([self._train, self._test], ignore_index=True)
        combinado = pd.get_dummies(combinado, columns=["weekday"], prefix="weekday")

        columnas_weekday = [c for c in combinado.columns if c.startswith("weekday_")]
        combinado[columnas_weekday] = combinado[columnas_weekday].astype(int)

        self._train = combinado.iloc[:n_train].reset_index(drop=True)
        self._test = combinado.iloc[n_train:].reset_index(drop=True)

    def obtener_datos(self):
        self._cargar_csv()
        self._codificar_weekday()

        X_train = self._train.drop(columns=[self._columna_objetivo])
        y_train = self._train[[self._columna_objetivo]]
        X_test = self._test.drop(columns=[self._columna_objetivo])
        y_test = self._test[[self._columna_objetivo]]

        return X_train, y_train, X_test, y_test


