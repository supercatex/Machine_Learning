import cv2 as cv
import random
import json
import numpy as np
from keras.models import Sequential
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import np_utils
from keras.datasets import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X:", X_train.shape, "y:", y_train.shape)
while True:
    rnd = random.randint(0, len(X_train))
    img = X_train[rnd].copy()
    img = cv.resize(img, (300, 300))
    cv.imshow("image", img)
    print(y_train[rnd])

    key = cv.waitKey(0)
    if key == 27:
        break
cv.destroyAllWindows()

X_train = X_train.reshape(len(X_train), 28, 28, 1)
print(X_train.shape)
X_train = X_train.astype('float32')
X_train /= 255
Y_train = np_utils.to_categorical(y_train, 10)

model = Sequential()
model.add(layers.Conv2D(
    input_shape=(28, 28, 1),
    filters=64,
    kernel_size=(5, 5),
    activation=activations.relu
))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation=activations.relu
))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation=activations.relu
))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation=activations.relu))
model.add(layers.Dense(512, activation=activations.relu))
model.add(layers.Dense(10, activation=activations.softmax))
model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy)

if os.path.exists("my_model.h5"):
    model.load_weights("my_model.h5")
else:
    model.fit(X_train, Y_train, epochs=10, batch_size=32)

    model.save_weights("my_model.h5")
    f = open("my_model.json", "w")
    f.write(model.to_json())
    f.close()


is_drawing = False
def onmouse_image(event, x, y, flags, param):
    global is_drawing, img
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing = True
    elif event == cv.EVENT_LBUTTONUP:
        is_drawing = False

    if event == cv.EVENT_MOUSEMOVE:
        if is_drawing:
            cv.circle(img, (x, y), 5, (255, 255, 255), -1)
            cv.imshow("image", img)


cv.namedWindow("image")
cv.setMouseCallback("image", onmouse_image)
img = np.zeros((300, 300))

while True:
    cv.imshow("image", img)
    key = cv.waitKey(0)

    if key == 32:
        tmp = img.copy()
        tmp = cv.resize(tmp, (28, 28))
        tmp = tmp.reshape(1, 28, 28, 1)
        tmp = tmp.astype('float32')
        tmp /= 255
        predict = model.predict(tmp)
        print(predict, np.argmax(predict))

        cv.waitKey(0)
        img = np.zeros((300, 300))

    if key == 27:
        break
cv.destroyAllWindows()
