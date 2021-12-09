import imp
from re import I
import tensorflow as tf
import pandas
import os
import numpy as np
import cv2
import json

import math


def projectPoints(xyz, K):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """

    xyz = np.transpose(xyz, (0, 2, 1))

    uv = np.matmul(K, xyz)
    uv = np.transpose(uv, (0, 2, 1))

    return uv[:, :, :2] / uv[:, :, -1:]


class MobileUNetDataGen(tf.keras.utils.Sequence):

    def __init__(self, image_directory,
                 label_directory,
                 label_k,
                 label_position,

                 batch_size=32,
                 input_size=(128, 128, 3),
                 shuffle=True):

        self.label_directory = label_directory
        self.image_directory = image_directory

        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        # creat file path array
        self.input_filepaths = np.sort(os.listdir(self.image_directory))

        fn_K_matrix = os.path.join(label_k)
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))

        fn_anno = os.path.join(label_position)
        with open(fn_anno, "r") as f:
            self.xyz = np.array(json.load(f))

        self.n = len(self.input_filepaths)

    def create_label_image(self, path):
        '''
            create label image for given image name
        '''
        None

    def load_label_image(self, name):
        '''
            load label image for given image name
            create a new one if it doesnt exist
        '''
        path = os.path.join(
            self.label_directory, name[0] + ".png")

        if not os.path.exists(path):
            self.create_label_image(name)

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr/255

        return image_arr

    def on_epoch_end(self):
        # if self.shuffle:
        #    self.df = self.df.sample(frac=1).reset_index(drop=True)
        None

    def get_input(self, name):
        '''Gets image and resize it for given image file name'''
        path = os.path.join(self.image_directory, name)
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])).numpy()

        return image_arr/255.

    '''
    
    
    
    '''

    def get_image_output(self, input):
        h = self.input_size[0]
        w = self.input_size[0]
        k = 1
        b = self.batch_size
        output = np.zeros([b, k, w, h])

        for i, (keypoints) in enumerate(input):
            kp_array = np.zeros([k, w, h])
            for j, (kp) in enumerate(keypoints):

                if j >= k:
                    break

                image = np.zeros([h, w])

                floorX = int(math.floor(kp[0]))
                floorY = int(math.floor(kp[1]))
                ceilX = int(math.ceil(kp[0]))
                ceilY = int(math.floor(kp[1]))

                floorX = floorX if floorX < 128 else 127
                floorY = floorY if floorY < 128 else 127
                ceilY = ceilY if ceilY < 128 else 127
                ceilX = ceilX if ceilX < 128 else 127

                image[floorX, floorY] = 1
                image[ceilX, ceilY] = 1
                image[floorX, ceilY] = 1
                image[ceilX, floorY] = 1
                image2 = cv2.GaussianBlur(image, (51, 51), 0)
                image += image2

                kp_array[j] = image

            output[i] = kp_array

        return output

    '''



    '''

    def get_output(self, index):

        l = len(self.K_matrix)

        sdx = index * self.batch_size % l
        edx = ((index + 1) * self.batch_size) % l

        if self.batch_size >= (l - sdx):
            sdx -= l
            if self.batch_size >= (l - edx):
                edx -= l

        Km = self.K_matrix[np.r_[sdx:edx]]
        xyz = self.xyz[np.r_[sdx:edx]]

        sps = projectPoints(xyz, Km) * 128/224
        output = self.get_image_output(sps)

        return output

    '''



    '''

    def get_input_data(self,  index):

        l = len(self.input_filepaths)
        sbx = index * self.batch_size % l
        ebx = (index + 1) * self.batch_size % l

        if self.batch_size >= (l - sbx):
            sbx -= l
            if self.batch_size >= (l - sbx):
                sbx -= l

        batches_filepaths = self.input_filepaths[np.r_[sbx: ebx]]

        x0_batch = np.asarray([self.get_input(x)
                               for x in batches_filepaths])
        return x0_batch

    def get_output_data(self,  index):

        y0_batch = self.get_output(index)

        return y0_batch

    def __getitem__(self, index):

        x = self.get_input_data(index)
        y = self.get_output_data(index)

        return x, y

    def __len__(self):
        return int(self.n / 32)
