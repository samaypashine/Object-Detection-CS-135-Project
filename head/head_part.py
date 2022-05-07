#!/usr/bin/env python
# coding: utf-8

# In[3]:



from constants import *
import tensorflow as tf
import tensorflow_addons as tfa


def preprocess_image(image):
    image = tf.image.decode_image(image)
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    image = tf.expand_dims(image, axis=0) / 255.0
    return image


def convolution_batchNormalization(inputs, filters, kernel_size, strides, padding="same", zero_pad=False, activation="leaky"):
    if zero_pad:
        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if activation == "leaky_relu":
        x = tf.keras.layers.LeakyReLU(alpha=ALPHA)(x)
    elif activation == "mish":
        x = tfa.activations.mish(x)

    return x

import tensorflow as tf

from tf2_yolov4.layers import conv_bn


def yolov4_head(input_shapes):

    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    x = conv_bn(input_3, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    

    maxpool_1 = tf.keras.layers.MaxPool2D((5, 5), strides=1, padding="same")(x)
    maxpool_2 = tf.keras.layers.MaxPool2D((9, 9), strides=1, padding="same")(x)
    maxpool_3 = tf.keras.layers.MaxPool2D((13, 13), strides=1, padding="same")(x)
    maxpool_4 = tf.keras.layers.MaxPool2D((26, 26), strides=1, padding="same")(x)
    maxpool_5 = tf.keras.layers.MaxPool2D((54, 54), strides=1, padding="same")(x)

    spp = tf.keras.layers.Concatenate()([maxpool_5,maxpool_4,maxpool_3, maxpool_2, maxpool_1, x])

    x = conv_bn(spp, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    
    output_3 = conv_bn(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(output_3, filters=256, kernel_size=1, strides=1, activation="leaky_relu")

    upsampled = tf.keras.layers.UpSampling2D()(x)

    x = conv_bn(input_2, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = tf.keras.layers.Concatenate()([x, upsampled])

    x = conv_bn(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    output_2 = conv_bn(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(output_2, filters=128, kernel_size=1, strides=1, activation="leaky_relu")

    upsampled = tf.keras.layers.UpSampling2D()(x)

    x = conv_bn(input_1, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = tf.keras.layers.Concatenate()([x, upsampled])

    x = conv_bn(x, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = conv_bn(x, filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    output_1 = conv_bn(x, filters=128, kernel_size=1, strides=1, activation="leaky_relu")

    return tf.keras.Model([input_1, input_2, input_3], [output_1, output_2, output_3], name="YOLOv4_head")


# In[ ]:




