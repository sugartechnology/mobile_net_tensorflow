import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


# load image file with its name
file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
# resize to 224*224
img = tf.keras.utils.load_img(file, target_size=[224, 224])

# load label file with its name
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# turn to tensor
x = tf.keras.utils.img_to_array(img)
# pre process it to use in mobile net
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis, ...])

# load pretrained model
#pretrained_model = tf.keras.applications.MobileNet()

mobilenet_save_path = '/var/folders/j4/rw33dpxs5934pb836nzz1p_r0000gn/T/tmp5uc0rzbc/mobilenet/1/'

loaded = tf.saved_model.load(mobilenet_save_path)
infer = loaded.signatures["serving_default"]

##print('pretrained model output names ', pretrained_model.output_names[0])
results = infer(tf.constant(x))
#print('results ', results)
labeling = results['predictions']

decoded = imagenet_labels[np.argsort(labeling)[0, ::-1][:5]+1]

print("Result after saving and loading:\n", decoded)
