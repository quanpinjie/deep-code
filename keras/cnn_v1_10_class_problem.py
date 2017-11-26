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
import keras
#from IPython.display import clear_output
import os
from tensorflow.python.client import device_lib
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


num_cores = 2 

num_CPU = 2 
num_GPU = 1 

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores,allow_soft_placement=True,\
        device_count={"CPU":num_CPU, "GPU": num_GPU})

session = tf.Session(config=config)
K.set_session(session)

def check_gpu_device():
    device = device_lib.list_local_devices()
    return device

print(check_gpu_device())

datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train,img_rows, img_cols, img_channels = train_features.shape
num_test, _, _, _ = test_features.shape
num_class = len(np.unique(train_labels))

print("train samples: {0}\ttest samples: {1}".format(num_train, num_test))

class_name = ["ariplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#fig = plt.figure(figsize=(8,3))
#
#for i in range(num_class):
#    ax = fig.add_subplot(2,5, 1 + i, xticks=[], yticks=[])
#    idx = np.where(train_labels[:] == i)[0][0]
#    features_idx = train_features[idx,::]
#    img_num = np.random.randint(features_idx.shape[0])
#    print("features: ", features_idx.shape)
#    im = np.transpose(features_idx)
#    #im = np.transpose(features_idx[img_num,::], (1,2,0))
#    ax.set_title(class_name[i])
#    plt.imshow(features_idx)
#plt.show()

# data pre-processing

train_features = (train_features.astype("float32") - train_features.mean())# / train_features.var()
test_features = (test_features.astype("float32") - test_features.mean())# / test_features.var()
print(train_labels[:10])
train_labels = np_utils.to_categorical(train_labels, num_class)
test_labels = np_utils.to_categorical(test_labels, num_class)
print(train_labels[:10])

# def function to plot accuracy

class Display(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.i = self.i + 1
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.var_losses, label="val_loss")
        plt.lengend()
        plt.show()

logdir = "./logs/{}".format("v1")

tensorBoard = keras.callbacks.TensorBoard(log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        write_images=True)

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    prdicted_class = np.argmax(result, axis=1)
    trun_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct) / result.shape[0]
    return (accuracy * 100)


def create_model():
    model = Sequential()
    model.add(Convolution2D(48,3,3,kernel_regularizer=regularizers.l2(0.01),border_mode="same",input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Convolution2D(48,3,3))
    #model.add(Convolution2D(48,3,3, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))
    model.add(Convolution2D(96,3,3))
    #model.add(Convolution2D(96,3,3, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    #model.add(Convolution2D(192,3,3, border_mode="same", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Convolution2D(192,3,3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Convolution2D(192,3,3))
    #model.add(Convolution2D(192,3,3,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(num_class, kernel_regularizer=regularizers.l2(0.01), activation="softmax"))

    adam = optimizers.Adam(lr=0.001, beta_1=0.6, beta_2=0.999, epsilon=1e-8, decay=1e-6)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

if __name__ == "__main__":
    model = create_model()
    print(model.summary())
    #print(model.get_config())
    dis = Display()
    #model_info = model.fit(train_features, train_labels, batch_size=128, nb_epoch=100,  validation_data=(test_features, test_labels), verbose=1, callbacks=[tensorBoard])
    #model_info = model.fit(train_features, train_labels, batch_size=256, nb_epoch=100,  validation_data=(test_features, test_labels), verbose=1)
    model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size=256),samples_per_epoch = train_features.shape[0],nb_epoch=100,  validation_data=(test_features, test_labels), verbose=1)
