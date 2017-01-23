#! /usr/bin/env python

import numpy as np
import pandas as pd
from utils import *
from matplotlib.pyplot import imshow

from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Create model for KerasClassifier  
def create_model(init_mode='uniform', learn_rate=0.1, momentum=0.3):
    model = Sequential()
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, init=init_mode, input_shape=(1, 28, 28), dim_ordering='th'))
    model.add(Activation("relu"))
    
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, init=init_mode, dim_ordering="th"))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Flatten())
    plot(model, to_file='model.png')
    
    # Add a fully-connected layer  
    model.add(Dense(output_dim=128, init=init_mode))
    model.add(Dropout(0.5))
    
    # Add another fully-connected layer with 10 neurons, one for each class.  
    model.add(Dense(output_dim=10, init=init_mode, activation="softmax"))

    # Compile model  
    sgd = SGD(lr=learn_rate, decay=0.0, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

# Fix random seed for reproducibility.  
seed = 7
np.random.seed(seed)

# Load the dataset  
num_classes = 10
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv').apply(lambda x: x/255.0)
X_train, y_train = train.ix[:,1:].apply(lambda x: x/255.0).as_matrix(), train.ix[:,0].as_matrix()
X_test = test.as_matrix()
print(X_train[0:15])
print(X_train.shape[0], X_train.shape[1])

# Rehape the data into 1x28x28 image.  
X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

print(X_train.shape)
# Categorize the label  
y_train = np_utils.to_categorical(y_train, num_classes)

# Create model  
model = KerasClassifier(build_fn=create_model, verbose=10)
# Define the grid search parameters  
batchs = [32]
epochs = [10, 20, 30, 40, 50]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'glorot_uniform']
#learn_rate = [0.001, 0.01, 0.1]
#momentum = [0.0, 0.3, 0.6, 0.9]
learn_rate = [0.1]
momentum = [0.3]

param_grid = dict(batch_size=batchs, nb_epoch=epochs, init_mode=init_mode, learn_rate=learn_rate, momentum=momentum)
clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=1)
clf_result = clf.fit(X_train, y_train)

# Summarize results  
print("Best: %f using %s" % (clf_result.best_score_, clf_result.best_params_))
means = clf_result.cv_results_['mean_test_score']
stds = clf_result.cv_results_['std_test_score']
params = clf_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

y_pred = clf.predict(X_test)
# Create submission  
submission = pd.DataFrame(index=pd.RangeIndex(start=1, stop=28001, step=1), columns=['Label'])
submission['Label'] = y_pred.reshape(-1, 1)
submission.index.name = "ImageId"
submission.to_csv('./submission.csv', index=True, header=True)
