from datetime import datetime
from re import M
import numpy as np
import tensorflow as tf
from mobile_net import mobile_net
from dataset_utils import read_data_set, read_test_image
from draw_utils import display
import os
import traceback
import logging
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims

import argparse
import sys

from mobile_net_dataset import MobileNetDataGen
from mobile_net_dataset import CreateMobileNetGent

import matplotlib.pyplot as plt


root_path = ''
save_path = os.path.join(root_path, 'mobilenet/2')
weights_path = os.path.join(root_path, 'mobilenet/2/weights')
label_path = os.path.join(root_path, 'Labels/labels.csv')
image_path = os.path.join(root_path, 'Images')
test_label_path = os.path.join(root_path, 'Labels/labels.csv')
test_image_path = os.path.join(root_path, 'Test')


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
            model = mobile_net((224, 224, 3), 11)
    
    elif weights_path is not None:
        model = mobile_net((224, 224, 3), 11)
        try:
            model.load_weights(weights_path)
            print('***************************')
            print('creating model from weights')
        except Exception as e:
            logging.error(traceback.format_exc())
            # Logs the error appropriately. :
            print('***************************')
            print('creating model from scratch')
    
    #adamOptimizer = tf.keras.optimizers.Adam(
    #        learning_rate=1e-4,
    #        name='Adam'
    #)
    #model.compile(optimizer=adamOptimizer, loss="mse",
    #                  metrics=["accuracy"])

    return model



def save_model(s_path, m_path):

    model = load_model(m_path)
    mobilenet_save_path = os.path.join(
        s_path, "mobilenet/2/")

    tf.saved_model.save(model, mobilenet_save_path)


def train_model(s_path, w_path,  l_path, batch_size=32, epochs=5):

    model = load_model(weights_path= w_path)

    ''''x, y = read_data_set(l_path)
    x = np.array([x])
    y = np.array([y])

    features_dataset = tf.data.Dataset.from_tensor_slices(x)
    labels_dataset = tf.data.Dataset.from_tensor_slices(y)

    data_sets = tf.data.Dataset.zip((features_dataset, labels_dataset))
    data_sets.batch(batch_size)
    data_sets.shuffle(5)'''

    data_sets = CreateMobileNetGent(l_path, batch_size=batch_size)

    checkpoint_filepath = w_path
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_freq=batch_size,
        save_best_only=True)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    '''for x, y in data_sets:  
        x1 = x * 255
        title = [x for x in range(batch_size)]
        display(x1, title, y)'''

    training_history = model.fit(
        data_sets,
        epochs=epochs,
        callbacks=[tensorboard_callback, model_checkpoint_callback])

    #model.save_weights(w_path)
    model.save(s_path)
    # model.summary()


def test_model(file_name):

    model = load_model(weights_path=weights_path)

    test_image = load_img(file_name, target_size=(224, 224))

    # convert the image to an array
    test_image = img_to_array(test_image)
    #print(img)
    test_image *= 1/255
    #print(img)
    # expand dimensions so that it represents a single 'sample'
    test_image = expand_dims(test_image, axis=0)
    results = model(test_image)

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





def create_arg_parser():
    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('command', help='command')
    parser.add_argument('--label_path', required=False)
    parser.add_argument('--image_path', required=False)
    parser.add_argument('--model_path', required=False)
    parser.add_argument('--save_path', required=False)
    parser.add_argument('--test_index', required=False)
    parser.add_argument('--epochs_num', required=False)
    parser.add_argument('--file_name', required=False)
    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])

    if parsed_args.command == 'train':
        l_path = parsed_args.label_path if parsed_args.label_path is not None else label_path
        i_path = parsed_args.image_path if parsed_args.image_path is not None else image_path
        w_path = parsed_args.model_path if parsed_args.model_path is not None else weights_path
        s_path = parsed_args.save_path if parsed_args.save_path is not None else save_path
        epocs = int(parsed_args.epochs_num) if parsed_args.epochs_num is not None else 50

        train_model(s_path, w_path,  l_path, epochs = epocs)
        
    elif parsed_args.command == 'test':
        test_model(parsed_args.file_name)
    
    elif parsed_args.command == 'display':
        test_image = read_test_image(os.path.join(test_image_path, '17750.jpg'))
        test_image = np.array([test_image])
        #labels = np.array([[1.0, 0.5115175,0.5042784,-0.04486451,0.03516448,0.0456779,-0.03639513,-0.0009806752,0.004245341,0.001017809,-0.00440630]])
        #labels = np.array([[1,0.4860586,0.4909233,0.07650164,0.02938423,-0.08001593,-0.0320636,0.008848786,0.01121643,-0.009032428,-0.0114491]])
        #labels = np.array([[1,0.5109575,0.4996452,0.06958157,0.03746924,-0.06275624,-0.03573748,-0.03004959,-0.004485548,0.03085494,0.0046057]])
        labels = np.array([[1,0.5333063,0.8620007,0.02647889,-0.04577672,-0.01820099,0.04923594,-0.05761877,0.001753569,0.06322676,-0.00192427]])
        
        display(test_image, ['predict'], labels)

    