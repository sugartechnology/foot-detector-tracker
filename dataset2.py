import tensorflow as tf
import os
import numpy as np
import cv2
import re
import math
import json

from indian_mobilnet_unet.jsonreader import (
    get_shoes_class_data,
    get_shoes_points_data,
    get_shoes_points_data_json,
)


class MobileDataSetGen2(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_directory,
        label_directory=None,
        file_range=(0, -1),
        batch_size=32,
        input_shape=(256, 256, 3),
        output_shape=(64, 64, 3),
        shuffle=True,
        name="train",
    ):
        self.image_directory = image_directory

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.shuffle = shuffle
        self.name = name

        self.file_range_start = file_range[0]
        self.file_range_end = file_range[1]

        self.label_directory = label_directory if label_directory else image_directory

        # creat file path array
        self.image_filepaths = np.sort(os.listdir(self.image_directory))
        self.image_filepaths = np.array(
            [k for k in self.image_filepaths if k.endswith("_image.jpg")]
        )

        self.image_filepaths = np.array(
            list(
                map(
                    lambda x: re.search("(.*\/)?(.*)(_image.jpg)$", x).groups()[1],
                    self.image_filepaths,
                )
            )
        )

        self.image_filepaths = self.image_filepaths[
            self.file_range_start : self.file_range_end
        ]

        self.jsons = []
        for fp in self.image_filepaths:
            f = open(self.label_directory + "/" + fp + ".json")
            self.jsons.append(json.load(f))
            f.close()
        self.jsons = np.array(self.jsons)

    def __getitem__(self, index):
        x = self.get_input_data(index, self.image_directory, self.image_filepaths)
        y = self.get_output_data(index, self.label_directory, self.image_filepaths)

        y_split = np.split(y, 10, 3)
        # y1 = np.split(y[0][0][0], 2, 2)
        # y_split = np.split(y[1][0][0], 8, 2)
        y = {
            "foot_mask": y_split[0],
            "leg_mask": y_split[1],
            "keypoints0": y_split[2],
            "keypoints1": y_split[3],
            "keypoints2": y_split[4],
            "keypoints3": y_split[5],
            "keypoints4": y_split[6],
            "keypoints5": y_split[7],
            "keypoints6": y_split[8],
            "keypoints7": y_split[9],
        }
        return (x, y)

    def __len__(self):
        return int(math.ceil(len(self.image_filepaths) / self.batch_size))

    #
    #
    #
    #
    def get_input(self, directory, name):
        """Gets image and resize it for given image file name"""
        path = os.path.join(directory, name + "_image.jpg")
        image = cv2.imread(path).astype(np.float32)
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        return image / 255.0

    #
    #
    #
    def get_output(self, directory, filePath, json):
        """Gets image and resize it for given image file name"""
        mask_path = directory + "/" + filePath + "_mask.jpg"
        # json_path = directory + "/" + filePath + ".json"

        image2 = get_shoes_class_data(mask_path)
        image2 = cv2.resize(image2, (self.output_shape[0], self.output_shape[0]))
        image3 = get_shoes_points_data_json(json)
        image3 = cv2.resize(image3, (self.output_shape[0], self.output_shape[0]))
        output = np.dstack((image2, image3))

        # print("mask path {} \n json path {}".format(mask_path, json_path))
        return output

    def get_input_data(self, index, directory, filepaths):
        filepaths = filepaths[np.r_[self.get_batchsiz_index(index)]]

        x0_batch = np.asarray([self.get_input(directory, x) for x in filepaths])
        return x0_batch

    def get_output_data(self, index, directory, filepaths):
        filePaths = filepaths[np.r_[self.get_batchsiz_index(index)]]
        jsons = self.jsons[self.get_batchsiz_index(index)]

        y0_batch = np.asarray(
            [
                self.get_output(directory, x, jsons[idx])
                for idx, x in enumerate(filePaths)
            ]
        )
        return y0_batch

    def get_batchsiz_index(self, index):
        l = len(self.image_filepaths)
        sbx = index * self.batch_size
        ebx = (index + 1) * self.batch_size

        ebx = l if ebx > l else ebx

        return range(sbx, ebx)


if __name__ == "__main__":
    t = MobileDataSetGen2(
        image_directory="/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData",
        batch_size=32,
        file_range=(0, 5),
    )

    for i, item in enumerate(t):
        for lm in item[1]["leg_mask"]:
            cv2.imshow("lm", lm)
            cv2.waitKey(0)

        print("-----")
