import cv2
from mobile_unet import load_model, weights_path
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims
import numpy as np
import math

# define a video capture object
#vid = cv2.VideoCapture(0)

model = load_model(weights_path=weights_path)

img = cv2.imread(
    '/Users/yufae/Downloads/FreiHAND_pub_v2_test/training/rgb/00000095.jpg')
# expand dimensions so that it represents a single 'sample'
img = cv2.resize(img, (128, 128))

img = expand_dims(img.astype("float64"), axis=0)

# prepare the image (e.g. scale pixel values for the vgg)
img *= 1/255

# get feature map for first hidden layer
results = model.predict(img)

a = results[0][0]
x0 = math.ceil(np.argmax(a)/128)
x1 = math.floor(np.argmax(a)/128)

y0 = np.argmax(a) % 128

a = results[0][0] * 5

img[0] = cv2.circle(img[0], (x0, y0), 2, (255, 0, 0))
cv2.imshow('result', a)

cv2.imshow('frame', img[0])

cv2.waitKey(-1)

# Destroy all the windows
cv2.destroyAllWindows()
