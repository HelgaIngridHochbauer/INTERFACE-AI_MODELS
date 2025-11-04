# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:03:53 2025

@author: Marc Frincu
"""

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

from pathlib import Path

import os

import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


number_of_pixels = 100
number_of_classes = 36

"""
Tests a provided model using images from the folder list.
Results are saved in the test_results inside a csv file.
"""
def test_model(folder_test_list, model_file):
    keras.backend.set_image_data_format('channels_first')
    model = create_ResNet50V2(number_of_pixels, number_of_classes)
    model.load_weights(model_file)
    
    test_images, file_list = get_test_data(folder_test_list)
    test_images = test_images.reshape(-1, 1, number_of_pixels, number_of_pixels)
    
    
    predictions = model.predict(test_images, use_multiprocessing=True)
      
    predictions = np.argmax(predictions, axis=1)
      
    
    df = pd.DataFrame([file_list, predictions]).transpose()
    
    np.savetxt('test_results/classes_' + str(number_of_classes) + "_prediction_" + Path(model_file).stem + "_model_.csv", df, delimiter="," , fmt='%s')
     
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

    
def resize(img, number_of_pixels):
    return cv2.resize(img, (number_of_pixels, number_of_pixels), interpolation=cv2.INTER_AREA)

def load_image(folder, name, number_of_pixels):
    image = cv2.imread(folder + f"{name:0>4}" + '.png', 0)
    print (folder + f"{name:0>4}" + '.png')
    return resize(image, number_of_pixels)    
    
def get_test_data(folder_list):
    
    images_test = np.empty(shape=(536, number_of_pixels, number_of_pixels), dtype=np.ubyte)
    file_list = []
    i = 0
    for dir in folder_list:
        print("Reading " + dir)
        for file in os. listdir(dir):
            print("Processing " + file)
            file_new = re.search('([0-9]+)', file).group(1)
            #os.rename(dir + file, dir + file_new + '.png')
            file_list.append(file_new)
            if os.path.isfile(dir + file):
                image = load_image(dir, file.removesuffix('.png'), number_of_pixels)
                images_test[i] = image
                print("Processed")
                i += 1
    return images_test, file_list

if (__name__ == '__main__'):  
    #72 classes with 5 deg rotated images 2025-08-19_RESNET_100px__100ep_0.41053786873817444acc.h5
    #72 classes with 1 deg rotated images 22_RESNET_100px_72cls__100ep_0.5927551984786987acc.h5
    #72 classes with 1 deg rotated images 2025-09-04_RESNET_100px_72cls__100ep_0.7098901271820068acc.h5
    test_model(['C:/Users/USER/datasets/pendants-bw/andreas/', 'C:/Users/USER/datasets/pendants-bw/maitane/'], 'C:/Users/USER/Documents/Research/AI4MultiGIS/Old/all_AS/arqueoastro/newnetangleclassif/models/2025-09-07_RESNET_100px_36cls__100ep_0.8296703100204468acc-bw.h5')