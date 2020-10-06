from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.callbacks import Callback

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

import tensorflow.keras.applications

from pathlib import Path

import cv2

from tf_explain.core.grad_cam import GradCAM

import math
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import DenseNet121,DenseNet169,  \
                                          DenseNet201,InceptionResNetV2,  \
                                          InceptionV3,MobileNet,MobileNetV2,  \
                                          NASNetLarge,NASNetMobile,ResNet101,  \
                                          ResNet101V2,ResNet152,ResNet152V2,  \
                                          ResNet50,ResNet50V2,VGG16,VGG19,Xception  

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


BATCH_SIZE = 8
IMG_HEIGHT = 512
IMG_WIDTH = 512
EPOCHS = 6

CHANNELS =3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)


lrs = [ ('6e-4',6e-4),('1e-3',1e-3), ('6e-3',6e-3)]

optimizers = ['Adam','Nadam','SGD']

architecture = ("ResNet152V2", ResNet152V2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet'))
                 


train_df = pd.read_csv("./train.csv")
valid_df = pd.read_csv("./val.csv")
test_df = pd.read_csv("./test.csv")

labels = ['fracture']

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, target_w = 512, target_h = 512):
    
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        height_shift_range=[-10,10],
        width_shift_range=[-10,10],
        rescale=1./255)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            target_size=(target_w,target_h))
    
    return generator

def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=2000, batch_size=8, seed=1, target_w = 512, target_h = 512):
    
    print("getting test and valid generators...")
    

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
       rescale=1./255)
    
    

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator

IMAGE_DIR = "/opt/AIStorage/PLAYGROUND/images/512/filtered/all_images"
train_generator = get_train_generator(train_df, IMAGE_DIR, "image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "image", labels)


def create_model():
    base_model = None
    model = None
    base_model = architecture[1]
        
    for layer in base_model.layers:
        if layer.name.endswith('bn'):
            layer.trainable = True
        else:
            layer.trainable = False
                
    x = base_model.output
        
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512,activation='relu')(x)
        
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(2, activation="sigmoid", dtype=tf.float32)(x)
        
    model = Model(inputs=base_model.input,outputs=predictions)
    
    return model


for l in lrs:
    for opt in  optimizers:
        lr = l[1]

        print('arch:', architecture[0])
        print('lr:', l[0])
        print('opt:', opt)
        
        STEPS_PER_EPOCH = len(train_generator) 
        
        VAL_STEPS_PER_EPOCH = len(valid_generator) 

        print("")
        print("Arch: ", architecture[0])
    
        
        optimizer = None
        
        if opt == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr)
        elif opt == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(lr=lr)
        else:
            optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)

        model = create_model()

        
        
        model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['acc'])
    
        history = model.fit(train_generator,
                    epochs=EPOCHS,
                    validation_data=valid_generator, 
                    steps_per_epoch=STEPS_PER_EPOCH, 
                    validation_steps=VAL_STEPS_PER_EPOCH,
                    callbacks=[])
        
        print(history.history)

        model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the model: ", len(model.layers))

        # Fine-tune from about the middel of the network
        fine_tune_at = int(0.5*len(model.layers))

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in model.layers[:fine_tune_at]:
            if layer.name.endswith('bn'):
                layer.trainable = True
            else:
                layer.trainable = False

        
        
        if opt == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr/6)
        elif opt == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(lr=lr/6)
        else:
            optimizer = tf.keras.optimizers.SGD(lr=lr/6, momentum=0.9, decay=1e-6, nesterov=True)

        model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['acc'])

        fine_tune_epochs = 8
        total_epochs =  EPOCHS + fine_tune_epochs

        history_fine = model.fit(train_generator,
                    epochs=total_epochs,
                    initial_epoch =  history.epoch[-1],
                    validation_data=valid_generator, 
                    steps_per_epoch=STEPS_PER_EPOCH, 
                    validation_steps=VAL_STEPS_PER_EPOCH,
                    callbacks=[])
        
        print(history_fine.history)

        model.trainable = True

        if opt == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr=lr/10)
        elif opt == 'Nadam':
            optimizer = tf.keras.optimizers.Nadam(lr=lr/10)
        else:
            optimizer = tf.keras.optimizers.SGD(lr=lr/10, momentum=0.9, decay=1e-6, nesterov=True)

        model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['acc'])

        unfreeze_epochs = 8
        total_epochs =  EPOCHS + fine_tune_epochs + unfreeze_epochs    

        history_unfreeze = model.fit(train_generator,
                    epochs=total_epochs,
                    initial_epoch =  history_fine.epoch[-1],
                    validation_data=valid_generator, 
                    steps_per_epoch=STEPS_PER_EPOCH, 
                    validation_steps=VAL_STEPS_PER_EPOCH,
                    callbacks=[])
        
        print(history_unfreeze.history) 

        model.evaluate(test_generator)

        model = None
        history = None
        history_fine = None
        history_unfreeze = None
