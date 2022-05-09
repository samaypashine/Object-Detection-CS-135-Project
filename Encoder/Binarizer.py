# /**
#  * @file Binarizer.py
#  * @author Samay Pashine, Lan Nguyen, Yuyao Guo, and Zhangcheng Sun
#  * @brief Encoding for RCNN - Object Detection
#  * @version 3.0
#  * @date 2022-05-09
#  * @copyright Copyright (c) 2022
#  */
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# Extension of the base class to encode the label for the binary classification.
class Label_Binarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
