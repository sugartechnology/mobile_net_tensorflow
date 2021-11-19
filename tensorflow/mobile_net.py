from tensorflow import keras
import tensorflow as tf


def my_IoU(y_pred, y_true):
    """
        IoU metric used in semantic segmentation.
    """
    inter = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    uni = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - inter
    return tf.reduce_mean(inter / uni)


def mobile_net_block(x, f, s=1):
    x = keras.layers.DepthwiseConv2D(3, strides=s, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    x = keras.layers.Conv2D(f, 1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def mobile_net(input_shape, num_classes):

    input = keras.Input(input_shape)
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    x = mobile_net_block(x, 64)

    x = mobile_net_block(x, 128, 2)
    x = mobile_net_block(x, 128)

    x = mobile_net_block(x, 256, 2)
    x = mobile_net_block(x, 256)

    x = mobile_net_block(x, 512, 2)
    for _ in range(5):
        x = mobile_net_block(x, 512)

    x = mobile_net_block(x, 1024, 2)
    x = mobile_net_block(x, 1024)

    x = keras.layers.GlobalAvgPool2D()(x)
    #x = keras.layers.Dropout(0.1)(x)

    output = keras.layers.Dense(num_classes, activation="linear")(x)
    model = keras.Model(input, output)

    # model = CustomModel()
    adamOptimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-6,
        name='Adam'
    )
    model.compile(optimizer=adamOptimizer, loss="mse", metrics=["accuracy"])
    return model
