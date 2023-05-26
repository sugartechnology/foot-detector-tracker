from datetime import datetime

import tensorflow as tf
from indian_mobilnet_unet.dataset import MobileDataSetGen
from indian_mobilnet_unet.dataset2 import MobileDataSetGen2
from indian_mobilnet_unet.mobileunet import (
    build_mobilenetv2_unet,
)
from indian_mobilnet_unet.mobileunet2 import build_mobilenetv2_unet2


DATA_GEN = MobileDataSetGen2
IMAGE_PATH = "/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData"
BATCH_SIZE = 32
MODEL_PATH = "multiple_output_256_64"
NET = build_mobilenetv2_unet2
EPOCHS = 100

if __name__ == "__main__":
    print(tf.__version__)

    model = NET()

    data_sets = DATA_GEN(
        image_directory=IMAGE_PATH,
        file_range=[0, 128],
        batch_size=BATCH_SIZE,
    )

    val_sets = DATA_GEN(
        image_directory=IMAGE_PATH,
        file_range=[128, 160],
        batch_size=BATCH_SIZE,
        name="val",
    )

    checkpoint_filepath = MODEL_PATH + "/check_point"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_freq=BATCH_SIZE * 10,
        save_best_only=True,
    )

    logdir = MODEL_PATH + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    try:
        training_history = model.fit(
            data_sets,
            epochs=EPOCHS,
            validation_data=val_sets,
            callbacks=[tensorboard_callback, model_checkpoint_callback],
        )
    except KeyboardInterrupt:
        model.save_weights(MODEL_PATH + "/weights")
        print('Output saved to: "{}./*"'.format(MODEL_PATH + "/weights"))

    model.save_weights(MODEL_PATH + "/weights")
