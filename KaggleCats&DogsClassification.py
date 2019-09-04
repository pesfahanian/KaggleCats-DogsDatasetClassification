import tensorflow as tf

# To make sure we are running the model on GPU.
# If your device doesn't have CUDA capble GPU, comment the line below.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

# Location of your test and train datasets on your local drive.
TRAIN_DATADIR = "Path to your training data"
TEST_DATADIR = "Path to your test data"
CATEGORIES = ["Dog", "Cat"]

# Resized image dimension.
IMG_SIZE = 50

training_data = []
test_data = []

def create_training_data(DATADIR):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if len(img_array) > 32 and len(img_array[0]) > 32:
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append(([new_array, class_num]))
                else:
                    continue
            except Exception as e:
                pass

def create_test_data(DATADIR):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if len(img_array) > 32 and len(img_array[0]) > 32:
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    test_data.append(([new_array, class_num]))
                else:
                    continue
            except Exception as e:
                pass

create_training_data(TRAIN_DATADIR)
create_test_data(TEST_DATADIR)
print(len(training_data), "training samples")
print(len(test_data), "test samples")

random.shuffle(training_data)
random.shuffle(test_data)

X_train = []
y_train = []

X_test = []
y_test = []

for features, labels in training_data:
    X_train.append(features)
    y_train.append(labels)
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_train = X_train/255.0
y_train = np.array(y_train)

for features, labels in test_data:
    X_test.append(features)
    y_test.append(labels)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test/255.0
y_test = np.array(y_test)

print(X_train.shape)
print(X_test.shape)

# Configure the model parameters.
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Configure the training parameters.
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
