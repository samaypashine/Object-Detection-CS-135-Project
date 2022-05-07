from fragments.constants import *
from fragments.auxiliary import *
from fragments.backbone import *
from fragments.neck import *
from fragments.head import *
import matplotlib.pyplot as plt


def YOLOv4(input_shape, num_classes, anchors, training=False, yolo_max_boxes=50, yolo_iou_threshold=0.5, yolo_score_threshold=0.5, weight_path=""):

    backbone = csp_darknet53(input_shape)
    neck = yolov4_neck(input_shapes=backbone.output_shape)
    normalized_anchors = compute_normalized_anchors(anchors, input_shape)
    head = yolov4_head(input_shapes=neck.output_shape, anchors=normalized_anchors, num_classes=num_classes, training=training, yolo_max_boxes=yolo_max_boxes, yolo_iou_threshold=yolo_iou_threshold, yolo_score_threshold=yolo_score_threshold)

    inputs = tf.keras.Input(shape=input_shape)
    lower_features = backbone(inputs)
    medium_features = neck(lower_features)
    upper_features = head(medium_features)

    yolov4 = tf.keras.Model(inputs=inputs, outputs=upper_features, name="YOLOv4")
    
    if weight_path != "":
        yolov4.load_weights(weight_path)
    return yolov4


if __name__ == "__main__":
    image_path = "./test.jpg"
    w_path = "./yolov4.h5"

    image = preprocess_image(image_path)
    model = YOLOv4(input_shape=(HEIGHT, WIDTH, 3), anchors=ANCHORS, num_classes=NUM_CLASSES, training=False, yolo_max_boxes=YOLO_MAX_BOXES, yolo_iou_threshold=YOLO_IOU_THRESHOLD, yolo_score_threshold=YOLO_SCORE_THRESHOLD, weight_path=w_path)

    model.summary()

    boxes, scores, classes, valid_detections = model.predict(image)
    boxes_ = boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
    classes_ = classes[0].astype(int)

    plt.figure(figsize=(16,10))
    plt.imshow(image[0])
    ax = plt.gca()

    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes_.tolist(), scores[0].tolist(), classes_.tolist()):
        if score > 0:
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=3))

          text = f'{CLASSES[cl]}: {score:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
