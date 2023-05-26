from datetime import datetime

import tensorflow as tf

from indian_mobilnet_unet.dataset3 import MobileDataSetGen3
from indian_mobilnet_unet.mubileunet3 import build_mobile_net_unet

DATA_GEN = MobileDataSetGen3
NET = build_mobile_net_unet
IMAGE_PATH = "/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData"
MODEL_PATH = "shoe_model_net_unet_huber"
BATCH_SIZE = 32
EPOCHS = 1000

if __name__ == "__main__":
    print(tf.__version__)

    model = NET(lr=0.0001)
    try:
        model.load_weights(MODEL_PATH + "/weights")
        print("load ok")
    except:
        print("load problem")

    data_sets = DATA_GEN(
        image_directory=IMAGE_PATH,
        # mask_directory=MASK_PATH,
        # label_file=LABEL_PATH,
        label_out_name="coordsOut",
        mask_out_name="classOut",
        file_range=[0, 32],
        point_counts=8,
        output_shape=(64, 64, 3),
        # batch_size=BATCH_SIZE,
        # input_shape=INPUT_SHAPE,
        # point_counts=POINT_COUNTS,
        # point_offset=POINT_OFFSET,
    )

    val_sets = DATA_GEN(
        image_directory=IMAGE_PATH,
        # mask_directory=MASK_PATH,
        # label_file=LABEL_PATH,
        point_counts=8,
        label_out_name="coordsOut",
        mask_out_name="classOut",
        file_range=[1000, 1100],
        output_shape=(64, 64, 3),
        # batch_size=BATCH_SIZE,
        # input_shape=INPUT_SHAPE,
        # point_counts=POINT_COUNTS,
        # point_offset=POINT_OFFSET,
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
