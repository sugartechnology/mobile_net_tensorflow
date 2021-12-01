import pandas
from PIL import Image
import numpy as np


def read_data_set(path):
    label_contents = pandas.read_csv(path)

    images_array = []
    labels_array = []

    for label in label_contents.values:

        image = Image.open(label[0].strip())
        images_array.append(np.array(image).dot(1/255))

        labels = []
        labels.append(label[1])
        labels.append(label[2])
        labels.append(label[3])
        labels.append(label[4])
        labels.append(label[5])
        labels.append(label[6])
        labels.append(label[7])
        labels.append(label[8])
        labels.append(label[9])
        labels.append(label[10])

        labels_array.append(labels)

    return images_array, labels_array


def read_test_image(path):

    image = Image.open(path)
    image_array = np.array(image).dot(1/255)

    print('image_array.shape', image_array.shape)

    return image_array
