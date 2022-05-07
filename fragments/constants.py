import numpy as np

EPOCHS = 20000

ALPHA = 0.1

IOU_THRESH = 0.6

HEIGHT = 640

WIDTH = 960

ANCHORS = [np.array([(12, 16), (19, 36), (40, 28)], np.float32),
           np.array([(36, 75), (76, 55), (72, 146)], np.float32),
           np.array([(142, 110), (192, 243), (459, 401)], np.float32)]

NUM_CLASSES = 40

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

YOLO_MAX_BOXES = 20

YOLO_IOU_THRESHOLD = 0.5

YOLO_SCORE_THRESHOLD = 0.8