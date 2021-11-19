from datetime import datetime
from re import M
import numpy as np
import tensorflow as tf
from mobile_net import mobile_net
from dataset_utils import read_data_set, read_test_image
from draw_utils import display
import os

from mobile_net_dataset import MobileNetDataGen
from mobile_net_dataset import CreateMobileNetGent


root_path = '/Users/yufae/Desktop/HandTraininDataSet/HandTrainingDataSet/Assets/Data2'
save_path = os.path.join(root_path, 'mobilenet/2')
model_path = os.path.join(root_path, 'Labels/model_weights')
label_path = os.path.join(root_path, 'Labels/labels.csv')
image_path = os.path.join(root_path, 'Images')
test_label_path = os.path.join(root_path, 'Labels/labels.csv')
test_image_path = os.path.join(root_path, 'Test')


def load_model(path):
    try:
        model = tf.saved_model.load(path)
        model = mobile_net((224, 224, 3), 10)

        adamOptimizer = tf.keras.optimizers.Adam(
            learning_rate=2e-6,
            name='Adam'
        )
        model.compile(optimizer=adamOptimizer, loss="mse",
                      metrics=["accuracy"])
    except:
        print('***************************')
        print('creating model from scratch')
        model = mobile_net((224, 224, 3), 10)

        adamOptimizer = tf.keras.optimizers.Adam(
            learning_rate=2e-6,
            name='Adam'
        )
        model.compile(optimizer=adamOptimizer, loss="mse",
                      metrics=["accuracy"])

    return model


def save_model(s_path, m_path):

    model = load_model(m_path)
    mobilenet_save_path = os.path.join(
        s_path, "mobilenet/2/")

    tf.saved_model.save(model, mobilenet_save_path)


def train_model(s_path, m_path, i_path,  l_path, batch_size=32, epochs=5):

    model = load_model('model/v1')

    ''''x, y = read_data_set(l_path)
    x = np.array([x])
    y = np.array([y])

    features_dataset = tf.data.Dataset.from_tensor_slices(x)
    labels_dataset = tf.data.Dataset.from_tensor_slices(y)

    data_sets = tf.data.Dataset.zip((features_dataset, labels_dataset))
    data_sets.batch(batch_size)
    data_sets.shuffle(5)'''

    data_sets = CreateMobileNetGent(l_path)

    checkpoint_filepath = 'model/v1'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        save_freq=10,
        save_best_only=True)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    training_history = model.fit(
        data_sets,
        epochs=epochs,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

    model.save_weights(m_path)
    tf.saved_model.save(model, s_path)
    # model.summary()


def test_model():

    model = load_model(model_path)
    test_image = read_test_image(
        os.path.join(test_image_path, '/2.jpg'))

    test_image = np.array([test_image])

    results = model.predict(test_image)

    display(test_image, ['test'], results)


def test_own(index):

    model = load_model(save_path)

    x, y = read_data_set(label_path)
    # turn to tensor
    test_image = tf.convert_to_tensor(
        [x[index], ], dtype=float
    )
    results = model(tf.constant(test_image))
    #display(test_image, ['real'], test_label)
    display(test_image, ['predict'], results)


train_model(save_path, model_path, image_path,  label_path, epochs=50)
# test_own(12)
# test_model()
# for i in range(25):
#    test_own(i)
