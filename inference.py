# /**
#  * @file inference.py
#  * @author Samay Pashine, Lan Nguyen, Yuyao Guo, and Zhangcheng Sun
#  * @brief Inference for RCNN - Object Detection with pre-trained weights.
#  * @version 3.0
#  * @date 2022-05-09
#  * @copyright Copyright (c) 2022
#  */

# Importing libraries
import os
import cv2
import tensorflow as tf
from train import RCNN

if __name__ == "__main__":
    
    images_path = input("[INTERACTIVE]. Enter the path to images : ")
    # images_path = "./test_images"
    Obj = RCNN(False)
    weight_path = input("[INTERACTIVE]. Enter the path for weight file : ")
    Obj.model = tf.keras.models.load_model(weight_path)
    
    if os.path.isdir(images_path):
        for img_path in os.listdir(images_path):
            img = cv2.imread(os.path.join(images_path, img_path))
            pred, coords = Obj.predict(img)

            for i in range(len(pred)):
                if pred[i][0][0] > 0.65:
                    cv2.rectangle(img, (coords[i][0], coords[i][1]), (coords[i][0] + coords[i][2], coords[i][1] + coords[i][3]), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("Image", img)
                    cv2.waitKey(100)
    else:
        if os.path.isfile(images_path):
            img = cv2.imread(os.path.join(images_path, images_path))
            pred, coords = Obj.predict(img)

            for i in range(len(pred)):
                if pred[i][0][0] > 0.65:
                    cv2.rectangle(img, (coords[i][0], coords[i][1]), (coords[i][0] + coords[i][2], coords[i][1] + coords[i][3]), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("Image", img)
                    cv2.waitKey(100)

    