
import matplotlib.pyplot as plt
from main_train import load_model, weights_path
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims


def visualize_layers(model):
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[1].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    plt.show()


def debug_layers(model):
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        #if 'conv' not in layer.name:
        #    continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)


def debug_feature_map(model, file):
    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    model.summary()
    # load the image with the required shape
    img = load_img(file, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    #print(img)

    img *= 1/255
    #print(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    #print(img)
    # prepare the image (e.g. scale pixel values for the vgg)
    #img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot all 64 maps in an 8x8 squares
    print(feature_maps.shape)
    square = 8
    ix = 1
    for _ in range(8):
        for _ in range(4):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()


def debug_feature_map_2(model, file):
    # redefine model to output right after the first hidden layer
    ixs = [2, 5, 9, 13, 17,79]
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)
    # load the image with the required shape
    img = load_img(file, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img *= 1/255
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot the output from each block
    square = 4
    for fmap in feature_maps:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        plt.show()

model = load_model(weights_path=weights_path)

debug_layers(model)
visualize_layers(model)
debug_feature_map_2(model, "D:\\Github\\mobile_net_tensorflow\\tensorflow\\Test\\28517.jpg")
