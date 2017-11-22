# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K

import time
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2017)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10


num_cores = 1

CPU=True
GPU=False
if GPU:
    num_GPU = 1
    num_CPU = 1

if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores,allow_soft_placement=True,\
        device_count={"CPU":num_CPU, "GPU": num_GPU})

session = tf.Session(config=config)
K.set_session(session)


(train_features, train_lables), (test_features, test_lablels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols = train_features.shape
num_test, _, _, _ = test_features.shape
num_class = len(np.unique(train_lables))
print("num_trains: {0}\timg_channels: {1}\timg_rows: {2}\timg_cols: {3}".format(num_train, img_channels, img_rows, img_cols))
