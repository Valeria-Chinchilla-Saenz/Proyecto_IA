import pandas as pd


class CargadorDatos:

    def __init__(
        self,
        ruta_train,
        ruta_test,
        columna_objetivo="cnt",
        columnas_excluir=None
    ):
        self._ruta_train = ruta_train
        self._ruta_test = ruta_test
        self._columna_objetivo = columna_objetivo
        self._columnas_excluir = columnas_excluir or []

        self._train = None
        self._test = None

    def _cargar_csv(self):
        self._train = pd.read_csv(self._ruta_train)
        self._test = pd.read_csv(self._ruta_test)

    def _codificar_weekday(self):
        # Combinar para mantener las mismas columnas dummy
        # tanto en train como en test.
        n_train = len(self._train)

        combinado = pd.concat(
            [self._train, self._test],
            ignore_index=True
        )

        combinado = pd.get_dummies(
            combinado,
            columns=["weekday"],
            prefix="weekday"
        )

        columnas_weekday = [
            c for c in combinado.columns
            if c.startswith("weekday_")
        ]

        combinado[columnas_weekday] = (
            combinado[columnas_weekday].astype(int)
        )

        self._train = (
            combinado.iloc[:n_train].reset_index(drop=True)
        )

        self._test = (
            combinado.iloc[n_train:].reset_index(drop=True)
        )

    def obtener_datos(self):
        self._cargar_csv()
        self._codificar_weekday()

        # Comprobar que la variable objetivo existe.
        for nombre, datos in [
            ("train", self._train),
            ("test", self._test)
        ]:
            if self._columna_objetivo not in datos.columns:
                raise ValueError(
                    f"La columna '{self._columna_objetivo}' "
                    f"no existe en {nombre}."
                )

        # Columnas que no deben entrar como predictoras.
        columnas_quitar = list(
            dict.fromkeys(
                [self._columna_objetivo]
                + self._columnas_excluir
            )
        )

        # Separar X e y.
        X_train = self._train.drop(
            columns=columnas_quitar,
            errors="ignore"
        )

        y_train = self._train[[self._columna_objetivo]]

        X_test = self._test.drop(
            columns=columnas_quitar,
            errors="ignore"
        )

        y_test = self._test[[self._columna_objetivo]]

        return X_train, y_train, X_test, y_test


