import os
import sys

import PIL.Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from tensorflow.keras import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# def plot_input_img(i):
#     plt.imshow(x_train[i], cmap='binary')
#     plt.title(y_train[i])
#     plt.show()
#
# for i in range(10):
#     plot_input_img(i)


# 1: preprocess

# normalizing to [0,1] range
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# add a dimension (z axis) so shape is (28,28,1)
x_train = np.expand_dims(x_train, -1)
print(x_train)
x_test = np.expand_dims(x_test, -1)


# change tags to categorical, which for some reason is called one hot vectors: 4 => to_categorical => [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

classes = 10
# 2: build the model

# linear set of layers
# model = tf.python.keras.models.sequential
# model = keras.models.Sequential()
#
# # create the first conv layer with 32 filters and a 3X3 kernel. output = 2d array of values = pixel values * kernel values
# model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
#
# # take max values from filters in a 2X2 grid. output = 2d array of values = max value from each 2X2 grid from each filter
# model.add(MaxPool2D((2, 2)))
#
# # duplicating the same process again
# model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
#
# # take the final values
# model.add(MaxPool2D((2, 2)))
#
# # convert to 1d array
# model.add(Flatten())
#
# # make sure CNN network can't relay on certain neurons and get fixated on certain placements
# model.add(Dropout(0.5))
#
# # classify a certain number as output
# model.add(Dense(classes, activation='softmax'))
#
# # compile the model
# model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#
# # get the behavior for each layer for development
# model.summary()
#
# # callbacks
# from keras.callbacks import EarlyStopping, ModelCheckpoint
#
# # EarlyStopping: if loss is not improve stop training and prevent ovefitting  along with Dropout
#
# ers = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)
#
# # ModelCheckpoint: save the best version of CNN network
# moc = ModelCheckpoint("./bestmodel.h5", monitor='val_accuracy', verbose=1, save_best_only=True)
#
# # add callbacks into an array which will be implemented in training
# cab = [ers, moc]
#
# # training model with x_train and y_train, 50 epochs or until early stopping
# his = model.fit(x_train, y_train, epochs=50, validation_split=0.3, callbacks=cab)

modelS = keras.models.load_model("C://Users//Yonatan//PycharmProjects//Handwriting2//bestmodel.h5")
#
#
evaluate = modelS.evaluate(x_test, y_test)
#
# print(f"model accuracy is {evaluate[1]}")
# print(f"model loss is {evaluate[0]}")
#
import PIL
count = 1
while os.path.isfile(f"numbers/m{count}.png"):
    image = PIL.Image.open(f"numbers/m{count}.png")
    image = PIL.ImageOps.grayscale(image)
    image = PIL.ImageOps.invert(image)
     # image.show()
     # image = np.resize(image)
    img = np.array(image)
    img = img.astype(np.float32)/255
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    prediction = np.argmax(modelS.predict(img))
    print(prediction)
    image.show()
    count += 1

print(np.shape(img))


img = cv2.imread("C://Users//Yonatan//PycharmProjects//Handwriting2//numbers//8.png", 0)
img = cv2.resize(img, (28, 28))
img = np.pad(img, (10, 10), 'constant', constant_values=0)
img = cv2.resize(img, (28, 28))/255

prediction = modelS.predict(img.reshape(1, 28, 28, 1))
label = np.argmax(prediction)

print(f"number is probably a {label}")






