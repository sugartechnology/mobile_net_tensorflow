import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# load image file with its name
file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
# resize to 224*224
img = tf.keras.utils.load_img(file, target_size=[224, 224])
# show it
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# turn to tensor
x = tf.keras.utils.img_to_array(img)
# pre process it to use in mobile net
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis, ...])


# load label file with its name
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

print(imagenet_labels)


# load pretrained model
pretrained_model = tf.keras.applications.MobileNet()
# predict
result_before_save = pretrained_model(x)
#
print("results before save")
print(result_before_save)


decoded = imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5]+1]
print("Result before saving:\n", decoded)


# save pretrained model
mobilenet_save_path = os.path.join(tmpdir, "mobilenet/1/")
print('saving pretrained model...')
print(mobilenet_save_path)
tf.saved_model.save(pretrained_model, mobilenet_save_path)
