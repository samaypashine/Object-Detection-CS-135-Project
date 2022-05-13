# Object-Detection-CS-135-Project
By **Samay Pashine**, **Lan Nguyen**, **Yuyao Guo** and **Zhangchen Sun** 

**Object Detection with Region-Based Convolution Neural Network based on VGG Backbone** is the project that we built R-CNN model based VGG backbone using python
and deep learning libraries like tensorflow, scikit-learn with airplane dataset.

![](https://miro.medium.com/max/1000/1*NLnnf_M4Nlm4p1GAWrWUCQ.gif)

## Introduction

### 1. Motivation
**Object detection** is a computer vision technique that allows us to identify and locate objects in an image or video. 

Our vision aims to utilize the R-CNN, which uses bounding boxes across the object regions to evaluate convolutional networks independently on all the region proposals and in turn classify multiple image regions into the proposed class. We further extend R-CNN by leveraging self-attention mechanism to enable the model to focus on discriminative parts on region proposals. We have faith that our application will **advance the object detection accuracy** through large-scale screening.

Please follow this [link](https://www.kaggle.com/datasets/pranavraikokte/airplanes-dataset-for-rcnn) to download our used dataset.

### 2.Project goals
- Read the source and do some **research** to understand more about the **dataset and its topic**.
- **Study to build R-CNN model in python from scratch**.
- **Study about the OpenCV functionalities**
- Perform **experiments with and without Attention Mechanism** on the R-CNN model based VGG backbone. 
- **Analyze the accuracy rate** more deeply and extract insights.
- **Visualize analysis** on **Model Loss graphs**.
### 3. Methods
* **Python** and some neccessary libraries such **Tensorflow, keras, OpenCV, scikit-learn**.
* **Visual Studio Code** to train models.

### 4. Building Models
* Building **R-CNN model**

The architecture is built by Tensorflow. More details can be found in `train.py`.

```python
        self.base_model = VGG16(weights='imagenet', include_top=True)
        self.freeze_layer(base_layer_train_flag)
        print("\n[INFO]. VGG-16 Model Summary : ")
        self.base_model.summary()

        X = self.base_model.layers[-2].output
        self.head = Dense(2, activation="softmax")(X)
        self.model = Model(self.base_model.input, self.head)
```

### 5. Model performance summary

Our model has the accuracy of **97.61 %** for the train dataset and **92.41 %** for the train dataset with attention layer in the model. 

## Conclusion

We successfully **built a deep neural network model** by implementing **Region-Based Convolutional Neural Network (R-CNN)** to automatically detect airplane image with high accuracy rate **97.61 %**.
In addition, we also **add Attention Mechanism** in our experiment for further advancement.
