import tensorflow as tf
import os
import numpy as np
import cv2
import re
import math

from indian_mobilnet_unet.jsonreader import get_shoes_class_data, get_shoes_points_data


class MobileDataSetGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_directory,
        label_directory=None,
        file_range=(0, -1),
        batch_size=32,
        input_shape=(256, 256, 3),
        shuffle=True,
        name="train",
    ):
        self.image_directory = image_directory

        self.batch_size = batch_size
        self.input_shape = input_shape
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

    def __getitem__(self, index):
        x = self.get_input_data(index, self.image_directory, self.image_filepaths)
        y = self.get_output_data(index, self.label_directory, self.image_filepaths)
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
        image = image / 255.0
        return image

    #
    #
    #
    def get_output(self, directory, filePath):
        """Gets image and resize it for given image file name"""
        mask_path = directory + "/" + filePath + "_mask.jpg"
        json_path = directory + "/" + filePath + ".json"

        image2 = get_shoes_class_data(mask_path)
        image3 = get_shoes_points_data(json_path)
        output = np.dstack((image2, image3))

        # print("mask path {} \n json path {}".format(mask_path, json_path))
        return output

    def get_input_data(self, index, directory, filepaths):
        filepaths = filepaths[np.r_[self.get_batchsiz_index(index)]]

        x0_batch = np.asarray([self.get_input(directory, x) for x in filepaths])
        return x0_batch

    def get_output_data(self, index, directory, filepaths):
        filePaths = filepaths[np.r_[self.get_batchsiz_index(index)]]

        y0_batch = np.asarray([self.get_output(directory, x) for x in filePaths])
        return y0_batch

    def get_batchsiz_index(self, index):
        l = len(self.image_filepaths)
        sbx = index * self.batch_size
        ebx = (index + 1) * self.batch_size

        ebx = l if ebx > l else ebx

        return range(sbx, ebx)


if __name__ == "__main__":
    t = MobileDataSetGen(
        image_directory="/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData",
        batch_size=32,
        file_range=(0, 50),
    )

    k = MobileDataSetGen(
        image_directory="/Users/yufae/Library/Application Support/DefaultCompany/ARSenteticData",
        batch_size=32,
        file_range=(50, 51),
    )

    for i, item in enumerate(t):
        print(item[0].shape)
        print(item[1].shape)
        print("-----")

    for i, item in enumerate(k):
        print(item[0].shape)
        print(item[1].shape)
        for sp in item[0]:
            cv2.imshow("ins", (sp * 255).astype(np.uint8))
            cv2.waitKey(0)

        for ins in item[1]:
            sps = np.split(ins, 10, 2)
            for sp in sps:
                cv2.imshow("ins", sp)
                cv2.waitKey(0)
        print("-----")
