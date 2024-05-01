import matplotlib
matplotlib.use('Agg')

import argparse
import os
import cv2
from imutils import paths
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from VGG import VGG


# Define a parameter parser
def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-train", "--data_train", default="training_1", help="path to input data_train")
    ap.add_argument("-test", "--data_test", default="training_1", help="path to input data_test")
    ap.add_argument("-m", "--model", default="cnn.model", help='path to output the model')
    ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss")
    args = vars(ap.parse_args())
    return args

#Parameter initialization
EPOCHS = 200
INIT_LR = 1e-3
BATCH_SIZES = 32
NUM_CLASS = 6
NORM_SIZE = 32

#Load the data
def load_data(path):
    print("begin to load iamges...")
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(250)
    random.shuffle(imagePaths)
    for ip in imagePaths:
        image = cv2.imread(ip)
        image = cv2.resize(image, (NORM_SIZE, NORM_SIZE))
        image = img_to_array(image)
        data.append(image)
        label = int(ip.split(os.path.sep)[-2])
        labels.append(label)

    #normalization
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    #The label is converted to a matte encoding form
    labels = to_categorical(labels, num_classes=NUM_CLASS)
    return data, labels

#Training function
def train(idg, X_train, X_test, y_train, y_test, args):
    print("compiling model ...")
    model = VGG.build(width=NORM_SIZE, height=NORM_SIZE, depth=3, classes=NUM_CLASS)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    print("......")
    print(len(X_train))
    #Training network parameter
    print("start to train network...")
    H = model.fit_generator(idg.flow(X_train, y_train, batch_size=BATCH_SIZES),
        validation_data=(X_test, y_test), steps_per_epoch=len(X_train)/BATCH_SIZES,
        epochs=EPOCHS, verbose=1)

    #Preservation network model
    print('save network...')
    model.save(args['model'])

    #Cost and accuracy of drawing training sets
    print("start to plot...")
    plt.style.use('ggplot')
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch#")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args['plot'])


if __name__=='__main__':
    args = args_parse()
    train_path = args['data_train']
    test_path = args['data_test']
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    #Data enhancement
    idg = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    train(idg, X_train, X_test, y_train, y_test, args)





