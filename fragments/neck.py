from constants import *
from auxiliary import *


def yolov4_neck(input_shapes):

    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    x = convolution_batchNormalization(input_3, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")

    maxpool_1 = tf.keras.layers.MaxPool2D((5, 5), strides=1, padding="same")(x)
    maxpool_2 = tf.keras.layers.MaxPool2D((9, 9), strides=1, padding="same")(x)
    maxpool_3 = tf.keras.layers.MaxPool2D((13, 13), strides=1, padding="same")(x)

    spp = tf.keras.layers.Concatenate()([maxpool_3, maxpool_2, maxpool_1, x])

    x = convolution_batchNormalization(spp, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    output_3 = convolution_batchNormalization(
        x, filters=512, kernel_size=1, strides=1, activation="leaky_relu"
    )
    x = convolution_batchNormalization(
        output_3, filters=256, kernel_size=1, strides=1, activation="leaky_relu"
    )

    upsampled = tf.keras.layers.UpSampling2D()(x)

    x = convolution_batchNormalization(input_2, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = tf.keras.layers.Concatenate()([x, upsampled])

    x = convolution_batchNormalization(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    output_2 = convolution_batchNormalization(
        x, filters=256, kernel_size=1, strides=1, activation="leaky_relu"
    )
    x = convolution_batchNormalization(
        output_2, filters=128, kernel_size=1, strides=1, activation="leaky_relu"
    )

    upsampled = tf.keras.layers.UpSampling2D()(x)

    x = convolution_batchNormalization(input_1, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = tf.keras.layers.Concatenate()([x, upsampled])

    x = convolution_batchNormalization(x, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=128, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=256, kernel_size=3, strides=1, activation="leaky_relu")
    output_1 = convolution_batchNormalization(
        x, filters=128, kernel_size=1, strides=1, activation="leaky_relu"
    )
    return tf.keras.Model([input_1, input_2, input_3], [output_1, output_2, output_3], name="YOLOv4_neck")

# output_1, output_2, output_3 = yolov4_neck([(52, 52, 256), (26, 26, 512), (13, 13, 1024)]).outputs
# if output_1.shape.as_list() == [None, 52, 52, 128]:
#     print(True)
# if output_2.shape.as_list() == [None, 26, 26, 256]:
#     print(True)
# if output_3.shape.as_list() == [None, 13, 13, 512]:
#     print(True)
