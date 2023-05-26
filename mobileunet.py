import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Conv2DTranspose,
    Concatenate,
    LeakyReLU,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

print("TF Version: ", tf.__version__)


def my_IoU(y_pred, y_true):
    i = y_pred * y_true
    u = y_pred * y_pred + y_true * y_true - i
    i = tf.reduce_sum(tf.reduce_sum(i, axis=-1), axis=-1)
    u = tf.reduce_sum(tf.reduce_sum(u, axis=-1), axis=-1)

    iou = (i + 1e-6) / (u + 1e-6)
    iou = tf.reduce_mean(iou)
    return iou


def my_IoU2(y_pred, y_true):
    """
    IoU metric used in semantic segmentation.
    """

    i = y_pred * y_true
    u = y_pred * y_pred + y_true * y_true - i
    i = tf.reduce_sum(i, axis=(1, 2))
    u = tf.reduce_sum(u, axis=(1, 2))

    iou = (i + 1e-6) / (u + 1e-6)
    iou = tf.reduce_mean(iou)
    return iou


def loss_IoU(y_pred, y_true):
    r = my_IoU(y_pred, y_true)
    return 1.0 - r


def loss_IoU2(y_pred, y_true):
    r = my_IoU2(y_pred, y_true)
    return 1.0 - r


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    return x


def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)

    return x


def build_mobilenetv2_unet(input_shape=(256, 256, 3)):  # (512, 512, 3)
    """Input"""
    inputs = Input(shape=input_shape)

    """ Pre-trained MobileNetV2 """
    encoder = MobileNetV2(
        include_top=False, weights=None, input_tensor=inputs, alpha=0.35
    )
    encoder.trainable = True

    """ Encoder """
    s1 = encoder.get_layer("input_1").output  # (512 x 512)
    s2 = encoder.get_layer("block_1_expand_relu").output  # (256 x 256)
    s3 = encoder.get_layer("block_3_expand_relu").output  # (128 x 128)
    s4 = encoder.get_layer("block_6_expand_relu").output  # (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer("block_13_expand_relu").output  # (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  # (64 x 64)
    d2 = decoder_block(d1, s3, 256)  # (128 x 128)
    d3 = decoder_block(d2, s2, 128)  # (256 x 256)
    d4 = decoder_block(d3, s1, 64)  # (512 x 512)

    outputs = tf.keras.layers.Conv2D(
        10,
        (3, 3),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
        padding="same",
        name="coordsOut",
        activation="relu",
    )(d4)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = tf.keras.Model(inputs=[inputs], outputs=outputs, name="MobileNetV2_U-Net")
    model.compile(
        optimizer=opt,
        loss=[loss_IoU2],
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_mobilenetv2_unet((256, 256, 3))
    model.summary()
