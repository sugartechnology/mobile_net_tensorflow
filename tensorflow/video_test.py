import cv2
from main_train import load_model, weights_path
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims

# define a video capture object
vid = cv2.VideoCapture(0)

model = load_model(weights_path=weights_path)

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    frame = frame[0:724, int((1280 - 724)/2):int((1280 - 724)/2 + 724), ...]
    frame = cv2.resize(frame, (224, 224))
    # Display the resulting frame

    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(frame.astype("float64"), axis=0)

    # prepare the image (e.g. scale pixel values for the vgg)
    img *= 1/255

    # get feature map for first hidden layer
    results = model.predict(img)

    # print(results)
    center_point = (int(results[0][1] * 224), 224-int(results[0][2] * 224))

    if results[0][0] > 0.96:
        cv2.circle(frame,
                   center_point,
                   radius=2,
                   color=(0, 0, 255),
                   thickness=1)

    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
