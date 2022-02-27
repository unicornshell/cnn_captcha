import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models, datasets, utils
import os
import numpy as np
from matplotlib import pyplot as plt


def CNNmodel(input_shape, filters=64, kernel=(3, 3), size=9, dropout=0.5, **kwargs):
    _inputs = layers.Input(shape=input_shape)  # 48， 48， 1
    x = layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(_inputs)
    x = layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(x)
    x = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)
    x = layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(_inputs)
    x = layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(x)
    x = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    x = layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(x)
    x = layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(x)
    x = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)
    x = layers.Dropout(0.25, name='dropout3')(x)
    x = layers.Conv2D(128, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)(x)
    x = layers.Conv2D(128, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)(x)
    x = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)
    x = layers.Dropout(0.25, name='dropout4')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(dropout, name='dropout')(x)

    x = layers.Dense(128, activation=tf.nn.relu)(x)
    x = layers.Dense(7, activation='softmax', name='FC')(x)
    return keras.Model(inputs=_inputs, outputs=x)


def preprocess_input(inputs, std=255., mean=0., expand_dims=None):
    inputs = tf.cast(inputs, tf.float32)
    inputs = (inputs - mean) / std
    # print(inputs.shape)
    if expand_dims is not None:
        np.expand_dims(inputs, expand_dims)
    # print(inputs.shape)
    return inputs


def img_aug_fun(elem):
    elem = tf.image.random_flip_left_right(elem)  # 左右翻转
    elem = tf.image.random_brightness(elem, max_delta=0.5)  # 调亮度
    elem = tf.image.random_contrast(elem, lower=0.5, upper=1.5)  # 调对比度
    elem = preprocess_input(elem)
    return elem


train = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\noname\Desktop\fer2013\Training", labels='inferred', label_mode='int',
    class_names=None, color_mode='grayscale', batch_size=32, image_size=(48, 48), shuffle=True, interpolation='bilinear'
)
# test=train=tf.keras.preprocessing.image_dataset_from_directory(
#  r"C:\Users\noname\Desktop\fer2013\PublicTest", labels='inferred',label_mode='int',
#  class_names=None, color_mode='grayscale', batch_size=32, image_size=(48,48), shuffle=True,interpolation='bilinear'
# )

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))
normalized_ds = test.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

model = CNNmodel(input_shape=(48,48,1),filters=64, kernel=(3,3),size=9)
model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train,validation_data=test,shuffle=True, batch_size=64, verbose=1,epochs=80)
model.save('my_model1.h5')

print("Evaluate on test data")
privtest=tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\noname\Desktop\fer2013\PrivateTest", labels='inferred',label_mode='int',
    class_names=None, color_mode='grayscale', batch_size=32, image_size=(48,48), interpolation='bilinear'
)
results = model.evaluate(privtest, batch_size=32)
print("test loss, test acc:", results)
