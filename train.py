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



if __name__ == "__main__":

    EPOCHS = 10
    STEP_PER_EPOCHS = 10
    VALIDATION_STEP = 2
    ALPHA = 1e-5
    NUM_CLASSES = 2
    TEST_SIZE = 0.10
    PATIENCE = 10
    images, labels = [], []
    images_path = "./dataset/Images"
    images_annotations = "./dataset/Airplanes_Annotations"

    cv2.setUseOptimized(True)
    sed_detector = cv2.createLineSegmentDetector()
    sed_detector = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    print("[INFO]. Segment Detector Instantiated.")

    images, labels = preprocessing(sed_detector, images_path, images_annotations)
    X, y = np.array(images), np.array(labels)
    print("[INFO]. Filtered the images and labels.")

    label_encoder = Label_Binarizer()
    y =  label_encoder.fit_transform(y)
    print("[INFO]. Encoding the labels.")


    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)

    print("[INFO]. X_train Shape : ", X_train.shape)
    print("[INFO]. Y_train Shape : ", y_train.shape)
    print("[INFO]. X_test Shape  : ", X_test.shape)
    print("[INFO]. Y_test Shape  : ", y_test.shape)


    train_data_gen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 90)
    train_data = train_data_gen.flow(x = X_train, y = y_train)

    test_data_gen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 90)
    test_data = test_data_gen.flow(x = X_test, y = y_test)
    print("[INFO]. Created the Data generator to include the variant of the images.")

    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)
    early = EarlyStopping(monitor='val_loss', min_delta = 0, patience = PATIENCE, verbose = 1, mode = 'auto')
    print("[INFO]. Checkpoint and Early stopping Mechanism created.")

    vgg_model = VGG16(weights = 'imagenet', include_top = True)
    print("\n[INFO]. VGG-16 Model Summary : ")
    vgg_model.summary()

    choice = input("[INTERACTIVE]. Do you want to train the VGG model as well? (Y/N) : ")
    if choice == 'y' or choice == 'Y':
        for layers in (vgg_model.layers)[:15]:
            layers.trainable = True
    else:
        for layers in (vgg_model.layers)[:15]:
            layers.trainable = False   

    middle_layer = vgg_model.layers[-2].output
    output = Dense(NUM_CLASSES, activation = "softmax")(middle_layer)

    print("[INFO]. Compling the Model.")
    model = Model(vgg_model.input, output)
    model.compile(loss = categorical_crossentropy, optimizer = Adam(lr=ALPHA), metrics=["accuracy"])
    model.summary()

    print("[INFO]. Starting the Training.")
    hist = model.fit(train_data, steps_per_epoch= STEP_PER_EPOCHS, epochs= EPOCHS, validation_data = test_data, validation_steps = VALIDATION_STEP, callbacks = [checkpoint, early])


    print("[INFO]. Plotting the loss graph.")
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["Loss","Validation Loss"])
    plt.show()

    print("[INFO]. Saving the loss graph at current directory.")
    plt.savefig('Loss_Graph.jpg')

