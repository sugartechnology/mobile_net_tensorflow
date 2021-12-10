from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization,  Conv2D, Input, Reshape,  MaxPool2D, ReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.merge import Concatenate
import logging
import traceback

from mobile_unet_dataset import MobileUNetDataGen

root_path = ''
weights_path = os.path.join(root_path, 'mobile_unet/weights')


def intersection_of_union(y_true, y_pred):

    i = y_pred * y_true

    u = y_pred * y_pred + y_true * y_true - i
    i = tf.reduce_sum(tf.reduce_sum(i, axis=-1), axis=-1)
    u = tf.reduce_sum(tf.reduce_sum(u, axis=-1), axis=-1)

    iou = (i + 1e-6) / (u + 1e-6)
    iou = tf.reduce_mean(iou)
    return iou


def mobile_unet_loss_function(y_true, y_pred):

    iou = intersection_of_union(y_true, y_pred)
    return 1-iou


def mobile_unet_acc_function(y_true, y_pred):

    iou = intersection_of_union(y_true, y_pred)
    return iou


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def decoder_block(inputs, skip_layer, num_filters):

    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_layer])
    x = conv_block(x, num_filters)

    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    x = MaxPool2D((2, 2), padding="same")(x)
    return x


def build_mobile_net_unet(input_shape):

    inputs = Input(shape=input_shape)

    model_neurons = 16
    e1 = encoder_block(inputs, model_neurons)
    e2 = encoder_block(e1, model_neurons * 2)
    e3 = encoder_block(e2, model_neurons * 4)

    # bottle_neck
    e4 = encoder_block(e3, model_neurons * 8)

    d1 = decoder_block(e4, e3, model_neurons * 8 + model_neurons * 4)
    d2 = decoder_block(d1, e2, model_neurons * 4 + model_neurons * 2)
    d3 = decoder_block(d2, e1, model_neurons * 2 + model_neurons * 1)

    output = UpSampling2D((2, 2))(d3)
    output = Conv2D(1, 1, padding="same", activation="sigmoid")(output)
    output = Reshape((1, 128, 128))(output)

    model = Model(inputs, output, name="Mobile_Unet")

    model.summary()

    # model = CustomModel()
    adamOptimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        name='Adam'
    )

    model.compile(optimizer=adamOptimizer,
                  loss=mobile_unet_loss_function, metrics=[mobile_unet_acc_function])

    return model


def load_model(save_path=None, weights_path=None):
    if save_path is not None:
        try:
            model = tf.saved_model.load(save_path)
            print(model)
            print('***************************')
            print('creating model from loading')
        except Exception as e:
            logging.error(traceback.format_exc())
            # Logs the error appropriately. :
            print('***************************')
            print('creating model from scratch')
            model = build_mobile_net_unet((128, 128, 3))

    elif weights_path is not None:
        model = build_mobile_net_unet((128, 128, 3))
        try:
            model.load_weights(weights_path)
            print('***************************')
            print('creating model from weights')
        except Exception as e:
            logging.error(traceback.format_exc())
            # Logs the error appropriately. :
            print('***************************')
            print('creating model from scratch')

    return model


if __name__ == "__main__":

    batch_size = 32
    epochs = 1000

    print("TF Version (0)", (tf.__version__))
    print("waihgts path", weights_path)
    model = load_model(weights_path=weights_path)

    data_sets = MobileUNetDataGen("/Users/yufae/Downloads/FreiHAND_pub_v2_test/training/rgb",
                                  "/Users/yufae/Downloads/FreiHAND_pub_v2_test/training",
                                  "/Users/yufae/Downloads/FreiHAND_pub_v2_test/training_K.json",
                                  "/Users/yufae/Downloads/FreiHAND_pub_v2_test/training_xyz.json",
                                  batch_size=batch_size)

    checkpoint_filepath = weights_path
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_freq=batch_size * 100,
        save_best_only=True)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    training_history = model.fit(
        data_sets,
        epochs=epochs,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

    model.save_weights(weights_path)
