from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Input, Reshape
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers.merge import Concatenate

from mobile_unet_dataset import MobileUNetDataGen


def mobile_unet_loss_function(y_pred, y_true):

    i = y_pred * y_true
    u = y_pred * y_pred + y_true * y_true - y_pred * y_true
    iou = (i + 1e-6)/(u + 1e-6)

    iou = tf.reduce_mean(iou)
    return 1-iou


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(inputs, skip_layer, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2,
                        padding="same")(inputs)
    x = Concatenate()([x, skip_layer])
    x = conv_block(x, num_filters)

    return x


def build_mobile_net_unet(input_shape):

    inputs = Input(shape=input_shape)
    # mobile_unet3 = MobileNetV3Small(alpha=.75, include_top=False,
    #                                weights='imagenet', input_tensor=input_tensor)

    # mobile_unet3.summary()

    mobile_unet2 = MobileNetV2(alpha=0.5, include_top=False,
                               weights=None, input_tensor=inputs)

    e1 = mobile_unet2.get_layer("input_1").output
    e2 = mobile_unet2.get_layer("block_1_expand_relu").output
    e3 = mobile_unet2.get_layer("block_3_expand_relu").output
    e4 = mobile_unet2.get_layer("block_6_expand_relu").output

    b = mobile_unet2.get_layer("block_13_expand_relu").output

    d1 = decoder_block(b, e4, 128)
    d2 = decoder_block(d1, e3, 64)
    d3 = decoder_block(d2, e2, 32)
    d4 = decoder_block(d3, e1, 16)

    output = Conv2D(21, 1, padding="same", activation="softmax")(d4)
    output = Reshape((21, 128, 128))(output)
    model = Model(inputs, output, name="Mobile_Unet")

    model.summary()

    # model = CustomModel()
    adamOptimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-6,
        name='Adam'
    )

    model.compile(optimizer=adamOptimizer,
                  loss=mobile_unet_loss_function, metrics=["accuracy"])

    return model


if __name__ == "__main__":

    root_path = ''
    weights_path = os.path.join(root_path, 'mobile_unet/weights')
    batch_size = 32
    epochs = 2

    print("TF Version (0)", (tf.__version__))
    model = build_mobile_net_unet((128, 128, 3))

    data_sets = MobileUNetDataGen("/Users/yufae/Downloads/FreiHAND_pub_v2/training/rgb",
                                  "/Users/yufae/Downloads/FreiHAND_pub_v2/training",
                                  "/Users/yufae/Downloads/FreiHAND_pub_v2/training_K.json",
                                  "/Users/yufae/Downloads/FreiHAND_pub_v2/training_xyz.json")

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
