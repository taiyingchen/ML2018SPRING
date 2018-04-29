# coding: utf-8
import numpy as np
import pandas as pd
import os
import sys

# Keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, LeakyReLU, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Constants
DIRECTORY = "ml-2018spring-hw3/"
MODEL_DIRECTORY = "model/"
LABEL_MAP = {0: '生氣', 1: '厭惡', 2: '恐懼', 3: '高興', 4: '難過', 5: '驚訝', 6: '中立'}

# Functions
def get_training_data(horizontal_flip=False, shuffle_data=False, validation_split=0.0):
    filepath = sys.argv[1]

    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        x_raw = data["feature"]
        y_raw = data["label"]
        
        #  Split features into array & reshape to (48, 48, 1)
        x = x_raw.str.split(expand=True).values.reshape(-1, 48, 48, 1).astype('int')
        # One hot encoding
        y = np_utils.to_categorical(y_raw)
        # Add fliplr image to label 1
        if horizontal_flip:
            (x, y) = add_fliplr_image(x, y, y_raw, 1)
        if shuffle_data:
            (x, y) = shuffle(x, y)
        
        # Split validation set
        if validation_split > 0.0 and validation_split <= 1.0:
            valid_size = int(validation_split*len(x))
            x_train = x[:-valid_size]
            x_valid = x[-valid_size:]
            y_train = y[:-valid_size]
            y_valid = y[-valid_size:]
        else:
            x_train = x
            y_train = y
            x_valid = []
            y_valid = []
    else:
        print("Error: No such file at %s" % filepath)

    return (x_train, y_train), (x_valid, y_valid), (x_raw, y_raw)
        
def output_prediction(y_test, filename="output.csv"):
    arr = [[i, int(y_test[i])] for i in range(len(y_test))]
    dw = pd.DataFrame(arr, columns = ["id", "label"])
    dw.to_csv(filename, index=False)

def add_fliplr_image(x_train, y_train, y_raw, label):
    index = y_raw[y_raw == label].index
    category = np_utils.to_categorical([label], 7)
    total_categories = np.repeat(category, len(index), axis=0)
    total_images = np.empty((0, 48, 48, 1), int)

    for i in index:
        image = np.fliplr(x_train[i]).reshape(1, 48, 48, 1)
        total_images = np.append(total_images, image, axis=0)

    x_train = np.concatenate((x_train, total_images), axis=0)
    y_train = np.concatenate((y_train, total_categories), axis=0)
    return (x_train, y_train)

def shuffle(x_train, y_train):
    seed = np.arange(x_train.shape[0])
    np.random.shuffle(seed)
    x_train = x_train[seed]
    y_train = y_train[seed]
    return (x_train, y_train)

def main():
    (x_train, y_train), (x_valid, y_valid), (x_raw, y_raw) = get_training_data(
        horizontal_flip=False,
        shuffle_data=False,
        validation_split=0.1)

    # Transform to 0 to 1
    x_train = x_train / 255
    if len(x_valid) > 0:
        x_valid = x_valid / 255

    # Normalization
    if len(x_valid) > 0:
        x_total = np.concatenate((x_train, x_valid))
    else:
        x_total = np.concatenate((x_train))
    mean = np.mean(x_total)
    std = np.std(x_total)

    x_train = (x_train - mean) / std
    if len(x_valid) > 0:
        x_valid = (x_valid - mean) / std

    # np.save("distribution.npy", [mean, std])

    # Image generator for data augmentation
    train_gen = ImageDataGenerator(
        zca_whitening=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    train_gen.fit(x_train)

    # Model configuration
    model = Sequential()

    # CNN
    model.add(Conv2D(64, 3, input_shape=(48, 48, 1), padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, 3, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, 3, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, 3, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, padding="same"))
    model.add(Dropout(0.2))

    model.add(Flatten())

    # DNN
    model.add(Dense(units=256, kernel_initializer="glorot_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=512, kernel_initializer="glorot_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(units=7,activation="softmax"))
    model.summary()

    # Checkpoint
    checkpoint_name = MODEL_DIRECTORY + "checkpoint.h5"
    checkpoint = ModelCheckpoint(checkpoint_name, monitor="val_acc", verbose=1, save_best_only=True, mode="max")

    # Training
    epochs = 100
    batch_size = 128
    steps_per_epoch = (x_train.shape[0]*5) // batch_size

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    train_history = model.fit_generator(
        train_gen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpoint])

if __name__ == "__main__":
    main()