# conda create --name keras_gpu python=3.7 keras-gpu tensorflow-gpu spyder
# conda activate keras_gpu
import tensorflow as tf
from keras.layers import Input

from datetime import date

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import backend as K

import tensorflow as tf

import tensorflow_addons as tfa

import csv
import cv2
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

number_of_pixels = 100
number_of_classes = 72

MODEL_SAVE_NAME = str(date.today()) + "_RESNET_" + str(number_of_pixels) + "px_"

# Keras ResNet50V2 model
def create_ResNet50V2(number_of_pixels, classes=3):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))


    return tf.keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )

def train_model(data_train, data_test, labels_train, labels_test):
    global MODEL_SAVE_NAME

    keras.backend.set_image_data_format('channels_first')
    model = create_ResNet50V2(number_of_pixels, number_of_classes)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        # optimizer=SGD(learning_rate=0.0001),
        # loss=tf.losses.BinaryCrossentropy(),
        loss=tf.losses.CategoricalCrossentropy(),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.losses.MeanSquaredError(),
        metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), tfa.metrics.F1Score(number_of_classes),
                 keras.metrics.AUC()]
    )

    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    epochs = 100
    result = model.fit(data_train,
                       labels_train,
                       epochs=epochs,
                       validation_data=(data_test, labels_test),
                       batch_size=16,
                       callbacks=[es]
                       # callbacks=[metrics]
                       # shuffle = True # optional parameter for composites only
                       )

    # original precision eval implementation
    loss, acc, prec, rec, f1, auc = model.evaluate(data_test, labels_test, verbose=1)

    # save model
    MODEL_SAVE_NAME += "_" + str(epochs) + "ep" + "_" + str(acc) + "acc"
    model.save('models/' + MODEL_SAVE_NAME + ".h5")

    print('\nTest loss:', loss)
    print('\nTest accuracy:', acc)
    print('\nMetric names:', model.metrics_names)

    keras.backend.clear_session()
    
def prepare_data_and_train(images, labels):
    # images /= 255
    labels = to_categorical(labels, number_of_classes)

    data_train, data_test, labels_train, labels_test = train_test_split(images,
                                                                        labels,
                                                                        test_size=0.1,
                                                                        # shuffle=True,
                                                                        # random_state=1
                                                                        shuffle=False,
                                                                        random_state=None
                                                                        )
    # validation_set_size = 300
    # data_validate = data_train[-validation_set_size:]
    # labels_validate = labels_train[-validation_set_size:]
    # data_train = data_train[:-validation_set_size]
    # labels_train = labels_train[:-validation_set_size]


    # reshape data for model compatibility
    data_train = data_train.reshape(-1, 1, number_of_pixels, number_of_pixels)
    # data_test_orig = np.copy(data_test)
    # data_validate_orig = np.copy(data_validate)
    data_test = data_test.reshape(-1, 1, number_of_pixels, number_of_pixels)
    # data_validate = data_validate.reshape(-1, 1, number_of_pixels, number_of_pixels)

    train_model(data_train, data_test, labels_train, labels_test)
    
def get_data(folder, number_of_pixels):
    row_count = 0
    df = pd.read_csv(folder + 'angles.txt',sep = ' ')
    row_count = df.shape[0]
    labels = []
    images = np.empty(shape=(row_count, number_of_pixels, number_of_pixels), dtype=np.ubyte)
    i = 0
    for index, row in df.iterrows():
      labels.append(get_class(row[1], 360 / number_of_classes))
      image = load_image(folder, row[0], number_of_pixels)
      images[i] = image
      i += 1

    images = np.resize(images, (row_count, number_of_pixels, number_of_pixels))
    return images, labels


def get_class(val, class_size = 1):
    return (int)(val // class_size)

def resize(img, number_of_pixels):
    return cv2.resize(img, (number_of_pixels, number_of_pixels), interpolation=cv2.INTER_AREA)

def load_image(folder, name, number_of_pixels):
    image = cv2.imread(folder + f"{name:0>4}" + '.png', 0)
    print (folder + f"{name:0>4}" + '.png')
    return resize(image, number_of_pixels)

if (__name__ == '__main__'):  
    images, labels = get_data('C:/Users/USER/datasets/rotation-angles/augmented/', number_of_pixels)
    prepare_data_and_train(images, labels)