import os
import cv2
import tensorflow as tf
from main import RCNN

if __name__ == "__main__":
    
    images_path = input("Enter the path to images : ")
    # images_path = "./test"
    Obj = RCNN(False)
    Obj.model = tf.keras.models.load_model("./model_weights.h5")
    
    if os.path.isdir(images_path):
        for img_path in os.listdir(images_path):
            img = cv2.imread(os.path.join(images_path, img_path))
            pred, coords = Obj.predict(img)

            for i in range(len(pred)):
                if pred[i][0][0] > 0.65:
                    cv2.rectangle(img, (coords[i][0], coords[i][1]), (coords[i][0] + coords[i][2], coords[i][1] + coords[i][3]), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow(img)
                    cv2.waitKey(0)
    else:
        if os.path.isfile(images_path):
            img = cv2.imread(os.path.join(images_path, images_path))
            pred, coords = Obj.predict(img)

            for i in range(len(pred)):
                if pred[i][0][0] > 0.65:
                    cv2.rectangle(img, (coords[i][0], coords[i][1]), (coords[i][0] + coords[i][2], coords[i][1] + coords[i][3]), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow(img)
                    cv2.waitKey(0)

    