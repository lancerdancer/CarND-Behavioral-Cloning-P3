import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

# import driving data from file
lines = []
with open('data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    image = cv2.imread(line[0])
    image_l = cv2.imread(line[1])
    image_r = cv2.imread(line[2])
    measurement = float(line[3])
    measurement_l = float(line[3]) + 0.2
    measurement_r = float(line[3]) - 0.2
    images.append(image)
    images.append(image_l)
    images.append(image_r)
    measurements.append(measurement)
    measurements.append(measurement_l)
    measurements.append(measurement_r)

# augment the dataset by flip all images and steering angle
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

# original images and labels
X_all = np.array(augmented_images)
y_all = np.array(augmented_measurements)
size_all = X_all.size
# shuffle all data
X_all, y_all = shuffle(X_all, y_all)
# use first 90% data as Training data, remaining 10% test
X_train = X_all[:int(size_all * 0.9)]
y_train = y_all[:int(size_all * 0.9)]
X_test = X_all[int(size_all * 0.9):]
y_test = y_all[int(size_all * 0.9):]

# Training model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# normalize x with zero mean 160x360x3
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# cropping image output 65x320x3
model.add(Cropping2D(cropping=((70,25),(0,0))))
# LeNet
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


#compile and fit the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')

# evaluate model against the test data
result = model.evaluate(X_test, y_test)
print('Scalar test loss:', result)
