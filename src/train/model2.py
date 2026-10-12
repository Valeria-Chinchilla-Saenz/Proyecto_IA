### Entrenamiento del modelo de clasificacion multiclase ###

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from src import config

class ModeloClasificacionDemanda:
    #Clase dedicada a la creación y el modelo de clasificación

    def __init__(self, x_train, y_train, num_capas, fraccion_apagadas, lotes = 256, epocas = 20):
        self._x_train = x_train
        self.num_capas = num_capas
        self.fraccion_apagadas = fraccion_apagadas
        self._y_train = y_train
        self.lotes = lotes
        self.epocas = epocas
        self.model = None

    def creacion_modelo(self):
        #Se construye y compila la arquitectura del modelo
        self.model = Sequential()
        num_neuronas = self._x_train.shape[1]

        num_neuronas_salida = self._y_train.iloc[:, 0].nunique()

        self.model.add(Input(shape=(num_neuronas,)))

        #Capas ocultas
        for capas in range(self.num_capas):
            self.model.add(Dense(units=num_neuronas, activation='relu'))
            self.model.add(Dropout(self.fraccion_apagadas))

        #Capa de salida clasificación multiclase
        self.model.add(Dense(num_neuronas_salida, activation='softmax'))

        #Copilación del modelo
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        return self.model

    def entrenamiento_modelo(self, x_train, y_train, x_val, y_val):
        #Definición de EarlyStopping para entrenar el modelo

        early_stop = EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose = 1,
            patience = 15)

        history = self.model.fit(
            x=x_train,
            y=y_train.values,
            epochs=self.epocas,
            batch_size=self.lotes,
            validation_data=(x_val, y_val),
            verbose=1,
            callbacks=[early_stop])

        return history

    def predecir(self, datos):
        prediccion = self.model.predict(datos)
        return prediccion

    def guardar_modelo(self):
        self.model.save(config.MODELS/"modelo2.keras")
        print('Modelo guardado')