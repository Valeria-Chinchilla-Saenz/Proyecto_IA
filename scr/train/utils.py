import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score, r2_score
from sklearn.metrics import accurancy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class EvaluacionPerdidas:
    def __init__(self, modelo):
        self._modelo = modelo
        self._losses = None

    def perdidas(self):
        losses = pd.DataFrame(self._modelo.history.history)
        self._losses = losses
        return losses

    def grafica_perdidas(self):
        grafico = self._losses.plot()
        return grafico


class Metricas:
    def __init__(self, y_test, predicciones):
        self._y_test = y_test
        self._predicciones =  predicciones
        self._mae = None
        self._mrse = None
        self._errores = None

    def mae(self):
        epa = mean_absolute_error(self._y_test, self._predicciones)
        self._mae = epa
        return epa

    def mrse(self):
        repc = np.sqrt(mean_squared_error(self._y_test, self._predicciones))
        self._mrse = repc
        return repc

    def varianza_explicada(self):
        ve = explained_variance_score(self._y_test, self._predicciones)
        return ve

    def r2(self):
        r2 = r2_score(self._y_test, self._predicciones)
        return r2

    def diferecia_porc_mae(self):
        return self._mae/self._y_test

    def diferencia_porc_mrse(self):
        return self._mrse/self._y_test

    def grafico_valorreal_prediccion(self):
        plt.scatter(self._y_test, self._predicciones)
        plt.xlabel("Valor Real")
        plt.ylabel("Predicción")

        plt.plot(self._y_test, self._y_test, 'r')

    def grafico_error(self):
        errores = self._y_test.values.reshape(self._y_test.shape[0], 1) - self._predicciones
        self._errores = errores
        sns.displot(errores)

    def grafico_errores_predicciones(self):
        sns.scatterplot(x=self._predicciones.flatten(), y=self._errores.flatten())
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicción")
        plt.ylabel("Error (real - predicción)")
        plt.title("Errores vs. Predicciones")


class MetricasClasificacion:
    def __init__(self, _y_test, y_predicciones):
        self._y_test = _y_test
        self.y_predicciones = y_predicciones

    def accurancy(self):
        #mide el procentaje total de predicciones correctas
        return accurancy_score(self._y_test, self.y_predicciones)

    def matriz_confusion(self):
        #muestra cuantas muestras fueron clasificadas correctamente o confundidas
        return confusion_matrix(self._y_test, self.y_predicciones)

    def reporte_clasificacion(self):
        #devuelve las métricas
        return classification_report(self._y_test, self.y_predicciones)

    def plot_matriz_confusion(self):
        cm = self.matriz_confusion()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Clase Predicha")
        plt.ylabel("Clase Real")
        plt.title("Matriz de Confusión")
        plt.show()











