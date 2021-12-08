import imp
import tensorflow as tf
import pandas
import os
import numpy as np


def CreateMobileNetGent(label_path, batch_size=32, target_size=(224, 224)):
    data_frame = pandas.read_csv(
        label_path)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1)

    df = image_generator.flow_from_dataframe(data_frame,
                                             directory=None,
                                             x_col='image',
                                             y_col=[
                                                 'c', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'],
                                             batch_size=batch_size,
                                             class_mode='raw',
                                             classes=None,
                                             shuffle=True,
                                             target_size=target_size)
    return df


class MobileNetDataGen(tf.keras.utils.Sequence):

    def __init__(self, image_directory, label_directory,
                 batch_size=32,
                 input_size=(224, 224, 3),
                 shuffle=True):

        self.label_directory = label_directory
        self.image_directory = image_directory

        self.data_frame = pandas.read_csv(
            label_directory)

        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)

        df = image_generator.flow_from_directory(self.data_frame,
                                                 directory=None,
                                                 x_col='image',
                                                 y_col=[
                                                     'c', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'],
                                                 batch_size=batch_size,
                                                 class_mode='raw',
                                                 classes=None,
                                                 shuffle=True,
                                                 target_size=(
                                                     input_size[0], input_size[1]),
                                                 )

        self.df = df

        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(
            image_arr, (self.input_size[0], self.input_size[1])).numpy()

        return image_arr/255.

    def __get_output(self, label):

        output = self.label_contents_dict[label]
        return output

    def __get_data(self, batches, batches_filpaths):
        # Generates data containing batch_size samples

        X_batch = (batches)

        y0_batch = np.asarray([self.__get_output(y)
                               for y in batches_filpaths])

        print('Y_batch.shape')
        print(y0_batch.shape)
        return X_batch, y0_batch

    def __getitem__(self, index):

        print(self.df.filepaths)
        print(self.df.__getitem__(index))

        batches = self.df.__getitem__(index)
        batches_filepaths = self.df.filepaths[index *
                                              self.batch_size: (index + 1) * self.batch_size]
        x, y = self.__get_data(batches, batches_filepaths)
        return x, y

    def __len__(self):
        return self.n


if __name__ == "__main__":

    label_path = "/Users/yufae/Desktop/mobile_net_tensorflow/mobile_net_tensorflow/tensorflow/test.csv"

    data_frame = pandas.read_csv(
        label_path)

    print(data_frame)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1)

    df = image_generator.flow_from_dataframe(data_frame,
                                             directory=None,
                                             x_col='image',
                                             y_col=[
                                                 'c'],
                                             batch_size=32,
                                             class_mode='raw',
                                             classes=None,
                                             shuffle=True,
                                             target_size=(224, 224))

    for i, item in enumerate(df):
        print(len(item[0]), len(item[1]), len(item[2]))
