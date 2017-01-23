#! /usr/bin/env python
from time import time

from utils import *
import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow

from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2, activity_l1l2

# Load the dataset  
t0 = time()
num_classes = 10
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv').apply(lambda x: x/255.0)
X_train, y_train = train.ix[:,1:].apply(lambda x: x/255.0).as_matrix(), train.ix[:, 0].as_matrix()
X_test = test.as_matrix()
print(X_train[0:15])

print(X_train.shape[0],X_train.shape[1])
# Rehape the data back into a 1x28x28 image  
X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

print(X_train.shape)
# Categorize the labels  
y_train = np_utils.to_categorical(y_train, num_classes)

# Design the model  
model = Sequential()

model.add(ZeroPadding2D(padding=(2, 2), input_shape=(1, 28, 28), dim_ordering='th'))
model.add(Convolution2D(64, 5, 5, dim_ordering='th'))
model.add(Activation("relu"))
model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
model.add(Convolution2D(128, 5, 5, dim_ordering='th'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
model.add(Convolution2D(256, 5, 5, dim_ordering='th'))
model.add(Activation("relu"))
model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(256, 3, 3, dim_ordering='th'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
model.add(Dropout(0.2))

model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, dim_ordering='th'))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(512, 3, 3, dim_ordering='th'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

model.add(Flatten())
plot(model, to_file='model.png')

model.add(Dropout(0.5))
# Add a fully-connected layer  
model.add(Dense(output_dim=2048, activation="relu"))

# Add another fully-connected layer with 10 neurons, one for each class.  
model.add(Dense(output_dim=10, W_regularizer=l2(0.01), activation="softmax"))
#model.add(Dense(output_dim=10, W_regularizer=l1l2(l1=0.01, l2=0.01), activity_regularizer=activity_l1l2(l1=0.01, l2=0.01), activation="softmax"))

# Compile the model  
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Data augmentation  
datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=0.1, # 0.2
        dim_ordering='th'
        #add_noise
    )

n_epoch = 10
seed = 1
image_folder = './image/'
for i in range(200): # Optimal is 40.  
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=256, seed=seed, save_to_dir=image_folder), len(X_train), nb_epoch=n_epoch, verbose=2)
    if (i+1)%5 == 0:
        filename = 'cycle' + str(i+1) + '_epoch' + str(n_epoch) + '_r15_s.5_l2new.csv' # cycle2_epoch10_s_wl1l2.csv  
        y_pred = model.predict_classes(X_test)
        np.savetxt(filename, np.c_[range(1, len(y_pred) + 1), y_pred], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
    # model.save('model_mnist.h5') # Uncomment to save your network.  
# model.fit(X_train, y_train, nb_epoch=10, batch_size=32)
# model.fit(X_train, y_train, nb_epoch=10, batch_size=32, validation_split=0.1)

dt = time() - t0
print('Used ' + str(dt) + ' seconds.')
