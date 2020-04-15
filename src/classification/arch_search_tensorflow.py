from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tensorflow.keras.callbacks import Callback

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

import tensorflow.keras.applications

from tensorflow.keras.applications import DenseNet121,DenseNet169,  \
                                          DenseNet201,InceptionResNetV2,  \
                                          InceptionV3,MobileNet,MobileNetV2,  \
                                          NASNetLarge,NASNetMobile,ResNet101,  \
                                          ResNet101V2,ResNet152,ResNet152V2,  \
                                          ResNet50,ResNet50V2,VGG16,VGG19,Xception  
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 64
IMG_HEIGHT = 512
IMG_WIDTH = 512
EPOCHS = 25
lr = 3e-3
CHANNELS =3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

FILTER_SIZE = 3

lrs = [1e-4, 3e-3, 1e-2]
batch_sizes = [16,32,64]

val_data_dir = '/opt/AIStorage/PLAYGROUND/images/1024/validation'
val_data_dir = pathlib.Path(val_data_dir)

data_dir = '/opt/AIStorage/PLAYGROUND/images/1024/train'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.png')))

val_image_count = len(list(val_data_dir.glob('*/*.png')))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

val_list_ds = tf.data.Dataset.list_files(str(val_data_dir/'*/*'))

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    label = -1
    if parts[-2] == 'negative':
        label = tf.constant([1.0, 0.0])
    else:
        label = tf.constant([0.0, 1.0])
    return label

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_labeled_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
VAL_STEPS_PER_EPOCH = np.ceil(val_image_count/BATCH_SIZE)

def prepare_for_training(ds, shuffle=True, cache=True, shuffle_buffer_size=11000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(labeled_ds)

valid_ds = prepare_for_training(val_labeled_ds, shuffle=False)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
with mirrored_strategy.scope():

    architectures = [("DenseNet121",DenseNet121(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("DenseNet169",DenseNet169(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("DenseNet201",DenseNet201(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("InceptionResNetV2",InceptionResNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("MobileNet",MobileNet(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("MobileNetV2",MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("ResNet101",ResNet101(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("ResNet101V2",ResNet101V2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("ResNet152",ResNet152(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("ResNet152V2",ResNet152V2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("ResNet50",ResNet50(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("ResNet50V2",ResNet50V2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("VGG16",VGG16(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("VGG19",VGG19(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')),
                 ("Xception",Xception(input_shape=IMG_SHAPE,include_top=False,weights='imagenet'))]
    
    for arch in architectures:
    
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print("")
        print("Arch: ",arch[0])
    
        initializer = tf.keras.initializers.he_normal()
        steps = np.ceil(image_count / BATCH_SIZE) * EPOCHS

        optimizer = tf.keras.optimizers.SGD(lr=lr, nesterov=True)
    
        #optimizer = tf.keras.optimizers.RMSprop(lr=lr)
        #optimizer = tf.keras.optimizers.Adam(lr=lr)
    
        base_model = arch[1]
    
        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        batch_norm_1 = tf.keras.layers.BatchNormalization()
        drop_out_1 = tf.keras.layers.Dropout(0.30)
        dense_layer_1 = tf.keras.layers.Dense(512,activation='relu', kernel_initializer=initializer)

        batch_norm_2 = tf.keras.layers.BatchNormalization()
        drop_out_2 = tf.keras.layers.Dropout(0.5)
        prediction_layer = tf.keras.layers.Dense(2)
    
        batch_norm_3 = tf.keras.layers.BatchNormalization()

        model = tf.keras.Sequential([
          base_model,
          global_average_layer,
      
          batch_norm_1,
          drop_out_1,
          dense_layer_1,
      
          batch_norm_2,
          drop_out_2,
          prediction_layer,
          batch_norm_3
        ])

    
        model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['acc'])
    
        #model.summary()
    
        history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=valid_ds, 
                    steps_per_epoch=STEPS_PER_EPOCH, 
                    validation_steps=VAL_STEPS_PER_EPOCH,
                    callbacks=[])
        print("")
        print("")

