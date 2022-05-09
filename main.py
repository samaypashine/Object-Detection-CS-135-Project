from tqdm import tqdm
from IOU.IOU import *
from Encoder.Binarizer import *
from bb_box.box import *
from preprocessing.code import *
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RCNN():
    def __init__(self, base_layer_train_flag = False, EPOCHS = 10, STEP_PER_EPOCHS = 10, VALIDATION_STEP = 2, ALPHA = 1e-5, NUM_CLASSES = 2, TEST_SIZE = 0.10, PATIENCE = 10, MIN_DELTA=0):
        self.EPOCHS = EPOCHS
        self.STEP_PER_EPOCHS = STEP_PER_EPOCHS
        self.VALIDATION_STEP = VALIDATION_STEP
        self.ALPHA = ALPHA
        self.NUM_CLASSES = NUM_CLASSES
        self.TEST_SIZE = TEST_SIZE
        self.PATIENCE = PATIENCE
        self.MIN_DELTA = MIN_DELTA

        self.base_model = VGG16(weights='imagenet', include_top=True)
        self.freeze_layer(base_layer_train_flag)
        print("\n[INFO]. VGG-16 Model Summary : ")
        self.base_model.summary()

        X = self.base_model.layers[-2].output
        self.head = Dense(2, activation="softmax")(X)
        self.model = Model(self.base_model.input, self.head)

        cv2.setUseOptimized(True)
        self.sed_detector = cv2.createLineSegmentDetector()
        self.sed_detector = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        print("[INFO]. Segment Detector Instantiated.")


    def freeze_layer(self, flag):
        for layers in (self.base_model.layers)[:15]:
            layers.trainable = flag


    def fit(self, train_data, test_data):
        print("[INFO]. Checkpoint and Early stopping Mechanism created.")
        checkpoint = ModelCheckpoint("./model_weights.h5", monitor='val_loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)
        early = EarlyStopping(monitor='val_loss', min_delta = self.MIN_DELTA, patience = self.PATIENCE, verbose = 1, mode = 'auto')
        
        print("[INFO]. Compling the Model.")
        self.model = Model(self.base_model.input, self.head)
        self.model.compile(loss = categorical_crossentropy, optimizer = Adam(lr = self.ALPHA), metrics = ["accuracy"])
        self.model.summary()

        print("[INFO]. Starting the Training.")
        self.hist = self.model.fit(train_data, steps_per_epoch = self.STEP_PER_EPOCHS, epochs = self.EPOCHS, validation_data = test_data, validation_steps = self.VALIDATION_STEP, callbacks = [checkpoint, early])
        return self.hist


    def predict(self, img, Acc_Threshold = 0.50):
        
        self.sed_detector.setBaseImage(img)
        self.sed_detector.switchToSelectiveSearchFast()
        results = self.sed_detector.process()
        imout = img.copy()
        
        preds, coords = [], []

        for e,result in enumerate(tqdm(results)):
            if e < 2000:
                x,y,w,h = result
                timage = imout[y:y+h,x:x+w]
                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out = self.model.predict(img)
                preds.append(out)
                coords.append([x, y, w, h])

        return preds, coords


if __name__ == "__main__":
    images, labels = [], []
    images_path = "./dataset/Images"
    images_annotations = "./dataset/Airplanes_Annotations"

    obj = RCNN(base_layer_train_flag = False)

    images, labels = preprocessing(obj.sed_detector, images_path, images_annotations)
    X, y = np.array(images), np.array(labels)
    print("[INFO]. Filtered the images and labels.")

    label_encoder = Label_Binarizer()
    y =  label_encoder.fit_transform(y)
    print("[INFO]. Encoding the labels.")

    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = obj.TEST_SIZE)
    print("[INFO]. X_train Shape : ", X_train.shape)
    print("[INFO]. Y_train Shape : ", y_train.shape)
    print("[INFO]. X_test Shape  : ", X_test.shape)
    print("[INFO]. Y_test Shape  : ", y_test.shape)

    train_data_gen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 90)
    train_data = train_data_gen.flow(x = X_train, y = y_train)

    test_data_gen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 90)
    test_data = test_data_gen.flow(x = X_test, y = y_test)
    print("[INFO]. Created the Data generator to include the variant of the images.")

    history = obj.fit(train_data, test_data)

    print("[INFO]. Plotting the loss graph.")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["Loss","Validation Loss"])
    plt.show()

    print("[INFO]. Saving the loss graph at current directory.")
    plt.savefig('Loss_Graph.jpg')
