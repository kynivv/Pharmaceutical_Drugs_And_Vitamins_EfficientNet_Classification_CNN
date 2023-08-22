# Libraries
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os

from tensorflow import keras
from keras import layers
from glob import glob
from zipfile import ZipFile
from keras.callbacks import ModelCheckpoint


# Extracting data from zip
with ZipFile('Pharmaceutical Drugs and Vitamins Dataset V2.zip') as zippfile:
    zippfile.extractall('dataset')


# Constants & Hyperparameters
BATCH_SIZE = 10
EPOCHS = 15
IMG_SIZE = 300
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
train_data_dir = 'dataset/Capsure Dataset/Train Image'
test_data_dir = 'dataset/Capsure Dataset/Test'


# Data Preparation
X_train = []
X_test = []
Y_train = []
Y_test = []

classes = os.listdir(train_data_dir)
num_of_classes = len(classes)

print(classes, num_of_classes)

data_dirs = (train_data_dir, test_data_dir)

for data_path in data_dirs:
    for i, name in enumerate(classes):
        print(f'{i} in {data_path}')
        images = glob(f'{data_path}/{name}/*.jpg')

        for image in images:
            img = cv2.imread(image)

            if data_path == train_data_dir:
                X_train.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
                Y_train.append(i)

            elif data_path == test_data_dir:
                X_test.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
                Y_test.append(i)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

Y_train = pd.get_dummies(Y_train)
Y_test = pd.get_dummies(Y_test)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# Model Based On EfficietNet
base_model = keras.applications.efficientnet.EfficientNetB3(include_top= False,
                                                            weights= 'imagenet',
                                                            input_shape= IMG_SHAPE,
                                                            pooling= 'max')

model = keras.Sequential([
    base_model,

    layers.BatchNormalization(),

    layers.Dense(256, activation= 'relu'),

    layers.Dropout(rate= 0.3),
    
    layers.Dense(20, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy']
              )


# Creating Model Callbacks
checkpoint = ModelCheckpoint('output/model_checkpoint.h5',
                             monitor= 'val_accuracy',
                             save_best_only= True,
                             save_weights_only= True,
                             verbose = 1
                             )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          validation_data= (X_test, Y_test),
          verbose= 1,
          callbacks= checkpoint,
          shuffle= True
          )