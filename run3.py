from datetime import datetime
from numpy import expand_dims

import tensorflow as tf
from indian_mobilnet_unet.dataset import MobileDataSetGen
from indian_mobilnet_unet.dataset2 import MobileDataSetGen2
from indian_mobilnet_unet.mobileunet import build_mobilenetv2_unet
import cv2
import numpy as np

from indian_mobilnet_unet.mobileunet2 import build_mobilenetv2_unet2
from indian_mobilnet_unet.mubileunet3 import build_mobile_net_unet


NET = build_mobilenetv2_unet2
IMAGE_PATH = "/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData"
MODEL_PATH = "multiple_output_256_64"
BATCH_SIZE = 32
EPOCHS = 50

if __name__ == "__main__":
    print(tf.__version__)

    model = NET()

    try:
        model.load_weights(MODEL_PATH + "/weights")
        print("load ok")
    except:
        print("load problem")

    img = cv2.imread(
        "/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData/2346_image.jpg"
    ).astype("float32")
    # expand dimensions so that it represents a single 'sample'
    # img /= 255.0
    # img = cv2.resize(img, (256, 128))
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)

    prediction = model.predict(img)

    i = 0
    for pred0 in prediction:
        # pp = np.split(pred0, 8, 2)
        for imsg in pred0:
            cv2.imshow("imm" + str(i), imsg / imsg.max())
            cv2.waitKey(0)
            i += 1
