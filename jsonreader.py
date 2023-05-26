import json
import time
import cv2
import numpy as np


def get_shoes_points_data(filePath):
    f = open(filePath)
    data = json.load(f)
    f.close()

    return get_shoes_points_data_json(data)


def get_shoes_points_data_json(data):
    image2 = None
    for i in range(8):
        x1 = int(data["shoes"][0]["points"][i]["x"] * 256)
        x2 = int(data["shoes"][1]["points"][i]["x"] * 256)

        y1 = 256 - int(data["shoes"][0]["points"][i]["y"] * 256)
        y2 = 256 - int(data["shoes"][1]["points"][i]["y"] * 256)

        imagetmp = np.zeros((256, 256, 1)).astype(np.float32)
        imagetmp = cv2.circle(imagetmp, (x1, y1), 5, (255, 255, 255), -1)
        imagetmp = cv2.circle(imagetmp, (x2, y2), 5, (255, 255, 255), -1)
        # print(imagetmp.max())
        imagetmp /= 255.0
        imagetmp = cv2.GaussianBlur(imagetmp, (7, 7), 3)

        # i = imagetmp[imagetmp > 0]
        # print(i)

        if image2 is not None:
            image2 = np.dstack((image2, imagetmp))
        else:
            image2 = imagetmp

    return image2


def get_shoe_class(x):
    # return math.sqrt(x)
    if x[2] > 0.5 and x[1] < 0.5:
        return 1.0
    return 0.0


def get_leg_class(x):
    # return math.sqrt(x)
    if x[1] > 0.5:
        return 1.0
    return 0.0


def get_shoes_class_data(filePath):
    image = cv2.imread(filePath)
    # image /= 255.0
    arr1 = np.where(image[..., 1] > 125, 255, 0).astype(np.uint8)
    arr2 = np.where((image[..., 1] < 125) & (image[..., 2] > 125), 255, 0).astype(
        np.uint8
    )
    # arr1 = np.apply_along_axis(get_shoe_class, 2, image)
    # arr2 = np.apply_along_axis(get_leg_class, 2, image)
    arr1 = np.dstack((arr1, arr2))
    # image /= 255.0
    return arr1 / 255


if __name__ == "__main__":
    image2 = get_shoes_class_data(
        "/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData/2310_mask.jpg"
    )
    image3 = get_shoes_points_data(
        "/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData/2310.json"
    )

    print(image2.shape)

    image2 = np.dstack((image2, image3))
    arr3 = np.array(np.dsplit(image2, 10))
    for imm in arr3:
        cv2.imshow("imm", imm)
        cv2.waitKey(0)
    print("-----------")

    get_shoes_class_data()
