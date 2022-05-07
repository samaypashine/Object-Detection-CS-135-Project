from constants import *
from auxiliary import *


def csp_block(inputs, filters, num_blocks):
    half_filters = filters // 2
    x = convolution_batchNormalization(inputs, filters=filters, kernel_size=3, strides=2, zero_pad=True, padding="valid", activation="mish")
    route = convolution_batchNormalization(x, filters=half_filters, kernel_size=1, strides=1, activation="mish")
    x = convolution_batchNormalization(x, filters=half_filters, kernel_size=1, strides=1, activation="mish")
    
    for _ in range(num_blocks):
        block_inputs = x
        x = convolution_batchNormalization(x, x.shape[3], kernel_size=1, strides=1, activation="mish")
        x = convolution_batchNormalization(x, x.shape[3], kernel_size=3, strides=1, activation="mish")
        x = x + block_inputs

    x = convolution_batchNormalization(x, filters=half_filters, kernel_size=1, strides=1, activation="mish")
    x = tf.keras.layers.Concatenate()([x, route])
    x = convolution_batchNormalization(x, filters=filters, kernel_size=1, strides=1, activation="mish")

    return x


def csp_darknet53(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = convolution_batchNormalization(inputs, filters=32, kernel_size=3, strides=1, activation="mish")
    x = convolution_batchNormalization(x, filters=64, kernel_size=3, strides=2, zero_pad=True, padding="valid", activation="mish",)
    
    route = convolution_batchNormalization(x, filters=64, kernel_size=1, strides=1, activation="mish")
    shortcut = convolution_batchNormalization(x, filters=64, kernel_size=1, strides=1, activation="mish")

    x = convolution_batchNormalization(shortcut, filters=32, kernel_size=1, strides=1, activation="mish")
    x = convolution_batchNormalization(x, filters=64, kernel_size=3, strides=1, activation="mish")
    x = x + shortcut

    x = convolution_batchNormalization(x, filters=64, kernel_size=1, strides=1, activation="mish")
    x = tf.keras.layers.Concatenate()([x, route])
    x = convolution_batchNormalization(x, filters=64, kernel_size=1, strides=1, activation="mish")

    x = csp_block(x, filters=128, num_blocks=2)
    output_1 = csp_block(x, filters=256, num_blocks=8)
    output_2 = csp_block(output_1, filters=512, num_blocks=8)
    output_3 = csp_block(output_2, filters=1024, num_blocks=4)

    return tf.keras.Model(inputs, [output_1, output_2, output_3], name="CSP_Darknet_53_layers")


# output_1, output_2, output_3 = csp_darknet53((416, 416, 3)).outputs
# if  output_1.shape.as_list() == [None, 52, 52, 256]:
#     print("TRUE")
# if  output_2.shape.as_list() == [None, 26, 26, 512]:
#     print("TRUE")
# if  output_3.shape.as_list() == [None, 13, 13, 1024]:
#     print("TRUE")
