import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

from indian_mobilnet_unet.mobileunet import decoder_block, loss_IoU

print("TF Version: ", tf.__version__)


def build_mobilenetv2_unet2(input_shape=(256, 256, 3)):  # (512, 512, 3)
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
    # d3 = decoder_block(d2, s2, 128)  # (256 x 256)
    # d4 = decoder_block(d3, s1, 64)  # (512 x 512)

    """ Output """
    outputs = []
    for i in range(10):
        if i == 0:
            outputs.append(
                tf.keras.layers.Conv2D(
                    1,
                    (3, 3),
                    padding="same",
                    name="foot_mask",
                    activation="sigmoid",
                )(d2)
            )
        elif i == 1:
            outputs.append(
                tf.keras.layers.Conv2D(
                    1,
                    (3, 3),
                    padding="same",
                    name="leg_mask",
                    activation="sigmoid",
                )(d2)
            )
        else:
            outputs.append(
                tf.keras.layers.Conv2D(
                    1,
                    (3, 3),
                    padding="same",
                    name="keypoints" + str(i - 2),
                    activation="sigmoid",
                )(d2)
            )
    # outputs.append(
    #    tf.keras.layers.Conv2D(2, (1, 1), activation="sigmoid", name="classOut")(d2)
    # )
    model = tf.keras.Model(inputs=[inputs], outputs=outputs, name="MobileNetV2_U-Net")

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss={
            "foot_mask": [loss_IoU],
            "leg_mask": [loss_IoU],
            "keypoints0": [loss_IoU],
            "keypoints1": [loss_IoU],
            "keypoints2": [loss_IoU],
            "keypoints3": [loss_IoU],
            "keypoints4": [loss_IoU],
            "keypoints5": [loss_IoU],
            "keypoints6": [loss_IoU],
            "keypoints7": [loss_IoU],
        },
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_mobilenetv2_unet2((256, 256, 3))
    model.summary()
