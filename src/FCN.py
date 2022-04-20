# -*- coding: utf-8 -*-

from __future__ import print_function

from tensorflow import keras
import numpy as np
import pandas as pd
from src.TimeHistory import TimeHistory


class FCN:
    def __init__(self, input_shape, nb_classes):
        self.model = self.build_model(input_shape, nb_classes)

    def build_model(self, input_shape, nb_classes):
        x = keras.layers.Input(input_shape)
        conv1 = keras.layers.Conv1D(128, 8, 1, padding='same')(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        conv2 = keras.layers.Conv1D(256, 5, 1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, 3, 1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        full = keras.layers.GlobalAveragePooling1D()(conv3)
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)

        model = keras.models.Model(inputs=x, outputs=out)

        # optimizer = keras.optimizers.Adam()
        optimizer = keras.optimizers.Nadam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def fit(self, x_train, x_test, Y_train, Y_test, nb_epochs=2000):
        batch_size = int(min(x_train.shape[0] / 10, 16))

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        hist = self.model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=0, validation_data=(x_test, Y_test), callbacks=[reduce_lr])
        # Print the testing results which has the lowest training loss.
        # log = pd.DataFrame(hist.history)
        # print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_accuracy'])
        log = pd.DataFrame(hist.history)
        acc = log.iloc[-1]['val_accuracy']
        return acc

    def fitAccLog(self, x_train, x_test, Y_train, Y_test, nb_epochs=2000):
        batch_size = int(min(x_train.shape[0] / 10, 16))

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        hist = self.model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=0, validation_data=(x_test, Y_test), callbacks=[reduce_lr])
        log = pd.DataFrame(hist.history)
        return log

    def fitTimeLog(self, x_train, x_test, Y_train, Y_test, nb_epochs=2000):
        batch_size = int(min(x_train.shape[0] / 10, 16))

        time_callback = TimeHistory()
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5, patience=50, min_lr=0.0001)
        hist = self.model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=0, validation_data=(x_test, Y_test), callbacks=[time_callback])
        log = pd.DataFrame(hist.history)
        df_time = pd.DataFrame(time_callback.times)
        return df_time

    '''
    def predict(self,x_test):
        y_pred=self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    '''
