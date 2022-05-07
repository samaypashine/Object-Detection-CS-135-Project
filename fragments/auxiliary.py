from constants import *

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
