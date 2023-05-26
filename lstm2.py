from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np
import pylab as plt


def build_lstm_net():
    seq = Sequential()

    seq.add(
        ConvLSTM2D(
            filters=10,
            kernel_size=(3, 3),
            input_shape=(None, 96, 96, 11),
            padding="same",
            return_sequences=True,
        )
    )
    seq.add(BatchNormalization())

    seq.add(
        ConvLSTM2D(
            filters=10, kernel_size=(3, 3), padding="same", return_sequences=True
        )
    )
    seq.add(BatchNormalization())

    seq.add(
        ConvLSTM2D(
            filters=10, kernel_size=(3, 3), padding="same", return_sequences=True
        )
    )
    seq.add(BatchNormalization())

    seq.add(
        ConvLSTM2D(
            filters=10, kernel_size=(3, 3), padding="same", return_sequences=True
        )
    )
    seq.add(BatchNormalization())

    seq.add(
        Conv3D(
            filters=10,
            kernel_size=(3, 3, 3),
            activation="sigmoid",
            padding="same",
            data_format="channels_last",
        )
    )
    seq.compile(loss="binary_crossentropy", optimizer="adadelta")
    seq.summary()
    return seq
