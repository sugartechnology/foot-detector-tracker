import json
import tensorflow as tf
import os
import numpy as np
import cv2
import re
import math
from PIL import Image


from indian_mobilnet_unet.jsonreader import (
    get_shoes_class_data,
    get_shoes_points_data,
    get_shoes_points_data_json,
)


class MobileDataSetGen3(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_directory,
        mask_directory="",
        label_file="",
        mask_out_name="mask_out_name",
        label_out_name="label_out_name",
        file_range=(0, -1),
        batch_size=32,
        input_shape=(128, 128, 3),
        output_shape=(128, 128, 3),
        point_counts=9,
        point_offset=0,
        shuffle=True,
        name="train",
    ):
        self.image_directory = image_directory

        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape if output_shape else self.input_shape
        self.shuffle = shuffle
        self.name = name
        self.label_out_name = label_out_name

        self.file_range_start = file_range[0]
        self.file_range_end = file_range[1]

        self.label_directory = mask_directory if mask_directory else image_directory

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
        return (x, {self.label_out_name: y})

    def __len__(self):
        return int(math.ceil(len(self.image_filepaths) / self.batch_size))

    ###
    def openWithPIL(self, path, width=128, height=128):
        image = Image.open(path)
        image = image.resize((width, height))
        return np.array(image)

    #
    #
    #
    #
    def get_input(self, directory, name):
        """Gets image and resize it for given image file name"""
        path = os.path.join(directory, name + "_image.jpg")
        image = cv2.imread(path)
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        return image

    #
    #
    #
    def get_output(self, directory, filePath, index, jsons):
        """Gets image and resize it for given image file name"""
        mask_path = directory + "/" + filePath + "_mask.jpg"
        # json_path = directory + "/" + filePath + ".json"

        # image2 = get_shoes_class_data(mask_path)
        # image2 = cv2.resize(image2, (self.input_shape[0], self.input_shape[1]))
        image3 = get_shoes_points_data_json(jsons[index])
        image3 = cv2.resize(image3, (self.output_shape[0], self.output_shape[1]))
        # output = np.dstack((image2, image3))
        # print(image3.shape)
        output = image3
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
                self.get_output(directory, x, idx, jsons)
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
    data = MobileDataSetGen3(
        image_directory="/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData",
        file_range=[1, 3],
        batch_size=32,
        label_out_name="coordsOut",
        mask_out_name="classOut",
        output_shape=(64, 64, 3),
    )

    print("******")
    for i, batch in enumerate(data):
        # print(i, batch[0].shape, batch[1]["classOut"].shape, batch[1]["coordsOut"].shape)
        # showObjectTransformation(batch[1]["classOut"][0], batch[1]["coordsOut"][0], batch[0][0], 1, False)
        # showObjectTransformation(batch[1]["classOut"][0], batch[1]["coordsOut"][0], batch[0][0], points_count=POINT_COUNTS, resolve_transformation=True)
        print("shape ", batch[1]["coordsOut"].shape)
        for bb in batch[0]:
            cv2.imshow("ss", bb)
            cv2.waitKey(0)

        for bb in batch[1]["coordsOut"]:
            sbbs = np.split(bb, 8, 2)
            for sbb in sbbs:
                cv2.imshow("ss", sbb)
                cv2.waitKey(0)
