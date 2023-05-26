import numpy as np
import tensorflow as tf
from torch import conv1d, conv2d
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    Conv2D,
    Conv2DTranspose,
    Concatenate,
    MaxPool2D,
    UpSampling2D,
)
from os import name
import tensorflow as tf
from tensorflow import keras
import numpy as np


def loss_iou(y_pred, y_true):
    y_i = y_pred * y_true
    y_u = y_pred * y_pred + y_true * y_true - y_i

    i = tf.reduce_sum(y_i, axis=(1, 2))
    u = tf.reduce_sum(y_u, axis=(1, 2))

    """tf.print(i)
    tf.print(u)
    tf.print(1.0-((i + 1e-6)/(u + 1e-6)))"""
    return tf.reduce_mean(1.0 - ((i + 1e-6) / (u + 1e-6)))


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    x = MaxPool2D((2, 2), padding="same")(x)
    return x


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)

    return x


def build_mobile_net_unet(
    input_shape=(256, 256, 3),
    modelName="fatih",
    lr=0.01,
    points_count=8,
    filter_count=16,
    drop_out=True,
    drop_out_k=1.0,
):
    xIn = tf.keras.Input(input_shape, dtype=np.dtype("uint8"))
    x = tf.keras.layers.Lambda(lambda x: x / 255)(xIn)

    e1 = encoder_block(x, filter_count)
    e2 = encoder_block(e1, filter_count * 2)
    e3 = encoder_block(e2, filter_count * 4)

    # bottle_neck
    e4 = encoder_block(e3, filter_count * 8)

    d1 = decoder_block(e4, e3, filter_count * 8 + filter_count * 4)
    # d2 = decoder_block(d1, e2, filter_count * 4 + filter_count * 2)
    # d3 = decoder_block(d2, e1, filter_count * 2 + filter_count * 1)

    output = UpSampling2D((3, 3))(d1)
    output = Conv2D(
        points_count, 1, padding="same", activation="sigmoid", name="coordsOut"
    )(output)

    outputs = []
    outputs.append(output)
    model = tf.keras.Model(inputs=[xIn], outputs=outputs, name=modelName)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss={"coordsOut": tf.keras.losses.Huber()},
        metrics={"coordsOut": ["mae", "mse"]},
    )

    return model


if __name__ == "__main__":
    model = build_mobile_net_unet()
    model.summary()
