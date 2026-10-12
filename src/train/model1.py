from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src import config

class ModeloRegresionDemanda:
    def __init__(self, x_train, y_train, numero_capas, fraccion_neuronas_apagadas, lotes, epocas,
                 early_stop=None):
        self._x_train = x_train
        self._y_train = y_train
        self._numero_capas = numero_capas
        self._fraccion_neuronas_apagadas = fraccion_neuronas_apagadas
        self._lotes = lotes
        self._epocas = epocas
        self._early_stop = early_stop or EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
        self._model = None

    def creacion_modelo(self):
        self._model = Sequential()

        numero_neuronas = self._x_train.shape[1]
        neuronas_salida_regresion = 1
        optimizador = 'adam'
        funcion_perdida = 'mse'

        for capas in range(self._numero_capas):
            self._model.add(Dense(units=numero_neuronas, activation='relu'))
            self._model.add(Dropout(self._fraccion_neuronas_apagadas))

        self._model.add(Dense(neuronas_salida_regresion))

        self._model.compile(optimizer=optimizador, loss=funcion_perdida)


    def entrenamiento_modelo(self):
        self._model.fit(x=self._x_train, y=self._y_train.values,
                   validation_split=0.2,
                   batch_size= self._lotes, epochs=self._epocas,
                   callbacks=[self._early_stop])
        return self._model


    def predecir(self, datos):
        predicion = self._model.predict(datos)
        return predicion

    def guardar_modelo(self):
        self._model.save(config.MODELS/"modelo1.keras")






