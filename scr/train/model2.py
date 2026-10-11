### Entrenamiento del modelo de clasificacion multiclase ###

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scr import config

class ModeloClasificacionDemanda:
    #Clase dedicada a la creación y el modelo de clasificación

    def __init__(self, x_train, num_capas, fraccion_apagadas, num_clases, lotes = 256, epocas = 20):
        self._x_train = x_train
        self.num_capas = num_capas
        self.fraccion_apagadas = fraccion_apagadas
        self.num_clases = num_clases
        self.lotes = lotes
        self.epocas = epocas
        self.model = None

    def creacion_modelo(self):
        #Se construye y compila la arquitectura del modelo
        self.model = Sequential()
        num_neuronas = self._x_train.shape[1]

        #Capas ocultas
        for capas in range(self.num_capas):
            if capas == 0:
                self.model.add(Dense(units=num_neuronas, activation='relu', input_shape=(num_neuronas,)))

            else:
                self.model.add(Dense(units=num_neuronas, activation='relu'))

            self.model.add(Dropout(self.fraccion_apagadas))

        #Capa de salida clasificación multiclase
        self.model.add(Dense(self.num_clases, activation='softmax'))

        #Copilación del modelo
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        return self.model

    def entramiento_modelo(self, x_train, y_train, x_val, y_val):
        #Definición de EarlyStopping para entrenar el modelo

        early_stop = EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose = 1,
            patience = 15)

        history = self.model.fit(
            x=x_train,
            y=y_train,
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