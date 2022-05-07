import numpy as np
from constants import *
from auxiliary import *


def yolov4_boxes_regression(feats_per_stage, anchors_per_stage):
    grid_size_x, grid_size_y = feats_per_stage.shape[1], feats_per_stage.shape[2]
    num_classes = feats_per_stage.shape[-1] - 5  # feats.shape[-1] = 4 + 1 + num_classes

    box_xy, box_wh, objectness, class_probs = tf.split(
        feats_per_stage, (2, 2, 1, num_classes), axis=-1
    )

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    grid = tf.meshgrid(tf.range(grid_size_y), tf.range(grid_size_x))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gy, gx, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.constant(
        [grid_size_y, grid_size_x], dtype=tf.float32
    )
    box_wh = tf.exp(box_wh) * anchors_per_stage

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs


def yolo_nms(yolo_feats, yolo_max_boxes, yolo_iou_threshold, yolo_score_threshold):
    bbox_per_stage, objectness_per_stage, class_probs_per_stage = [], [], []

    for stage_feats in yolo_feats:
        num_boxes = (
            stage_feats[0].shape[1] * stage_feats[0].shape[2] * stage_feats[0].shape[3]
        )  # num_anchors * grid_x * grid_y
        bbox_per_stage.append(
            tf.reshape(
                stage_feats[0],
                (tf.shape(stage_feats[0])[0], num_boxes, stage_feats[0].shape[-1]),
            )
        )  # [None,num_boxes,4]
        objectness_per_stage.append(
            tf.reshape(
                stage_feats[1],
                (tf.shape(stage_feats[1])[0], num_boxes, stage_feats[1].shape[-1]),
            )
        )  # [None,num_boxes,1]
        class_probs_per_stage.append(
            tf.reshape(
                stage_feats[2],
                (tf.shape(stage_feats[2])[0], num_boxes, stage_feats[2].shape[-1]),
            )
        )  # [None,num_boxes,num_classes]

    bbox = tf.concat(bbox_per_stage, axis=1)
    objectness = tf.concat(objectness_per_stage, axis=1)
    class_probs = tf.concat(class_probs_per_stage, axis=1)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(bbox, axis=2),
        scores=objectness * class_probs,
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
    )

    return [boxes, scores, classes, valid_detections]


def yolov4_head(input_shapes, anchors, num_classes, training, yolo_max_boxes, yolo_iou_threshold, yolo_score_threshold):
    input_1 = tf.keras.Input(shape=filter(None, input_shapes[0]))
    input_2 = tf.keras.Input(shape=filter(None, input_shapes[1]))
    input_3 = tf.keras.Input(shape=filter(None, input_shapes[2]))

    x = convolution_batchNormalization(input_1, filters=256, kernel_size=3, strides=1, activation="leaky_relu")

    x = tf.keras.layers.Conv2D(filters=len(anchors[0]) * (num_classes + 5), kernel_size=1, strides=1, padding="same", use_bias=True)(x)
    output_1 = tf.keras.layers.Reshape((x.shape[1], x.shape[2], len(anchors[0]), num_classes + 5))(x)

    x = convolution_batchNormalization(input_1, filters=256, kernel_size=3, strides=2, zero_pad=True, padding="valid", activation="leaky_relu")
    
    x = tf.keras.layers.Concatenate()([x, input_2])
    
    x = convolution_batchNormalization(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=512, kernel_size=3, strides=1, activation="leaky_relu")
    
    connection = convolution_batchNormalization(x, filters=256, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(connection, filters=512, kernel_size=3, strides=1, activation="leaky_relu")

    x = tf.keras.layers.Conv2D(filters=len(anchors[1]) * (num_classes + 5), kernel_size=1, strides=1, padding="same", use_bias=True)(x)
    output_2 = tf.keras.layers.Reshape((x.shape[1], x.shape[2], len(anchors[1]), num_classes + 5))(x)

    x = convolution_batchNormalization(connection, filters=512, kernel_size=3, strides=2, zero_pad=True, padding="valid", activation="leaky_relu",)
    
    x = tf.keras.layers.Concatenate()([x, input_3])

    x = convolution_batchNormalization(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=512, kernel_size=1, strides=1, activation="leaky_relu")
    x = convolution_batchNormalization(x, filters=1024, kernel_size=3, strides=1, activation="leaky_relu")

    x = tf.keras.layers.Conv2D(filters=len(anchors[2]) * (num_classes + 5), kernel_size=1, strides=1, padding="same", use_bias=True)(x)
    output_3 = tf.keras.layers.Reshape((x.shape[1], x.shape[2], len(anchors[2]), num_classes + 5))(x)

    if training:
        return tf.keras.Model([input_1, input_2, input_3], [output_1, output_2, output_3], name="YOLOv4_head")

    predictions_1 = tf.keras.layers.Lambda(lambda x_input: yolov4_boxes_regression(x_input, anchors[0]), name="yolov4_boxes_regression_small_scale")(output_1)
    predictions_2 = tf.keras.layers.Lambda(lambda x_input: yolov4_boxes_regression(x_input, anchors[1]), name="yolov4_boxes_regression_medium_scale")(output_2)
    predictions_3 = tf.keras.layers.Lambda(lambda x_input: yolov4_boxes_regression(x_input, anchors[2]), name="yolov4_boxes_regression_large_scale")(output_3)

    output = tf.keras.layers.Lambda(lambda x_input: yolo_nms(x_input, yolo_max_boxes=yolo_max_boxes, yolo_iou_threshold=yolo_iou_threshold, yolo_score_threshold=yolo_score_threshold), name="yolov4_nms")([predictions_1, predictions_2, predictions_3])

    return tf.keras.Model([input_1, input_2, input_3], output, name="YOLOv4_head")

bounding_box_shape = 4
objectness_score_shape = 1
expected_head_shape = (40 + objectness_score_shape) + bounding_box_shape

output_1, output_2, output_3 = yolov4_head([(52, 52, 128), (26, 26, 256), (13, 13, 512)], [np.array([(12, 16), (19, 36), (40, 28)], np.float32), np.array([(36, 75), (76, 55), (72, 146)], np.float32), np.array([(142, 110), (192, 243), (459, 401)], np.float32),], 40, training=True, yolo_max_boxes=20, yolo_iou_threshold=0.5, yolo_score_threshold=0.8).outputs
if output_1.shape.as_list() == [None, 52, 52, 3, expected_head_shape]:
    print(True)
if output_2.shape.as_list() == [None, 26, 26, 3, expected_head_shape]:
    print(True)
if output_3.shape.as_list() == [None, 13, 13, 3, expected_head_shape]:
    print(True)
