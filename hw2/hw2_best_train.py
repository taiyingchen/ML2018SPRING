
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os.path

# Numpy print full array
# np.set_printoptions(threshold=np.inf)

# Constants
DIRECTORY = "ntu-ml2018spring-hw2/"

# Parameters
norm_column = [0, 10, 78, 79, 80]

def extract_feature(x):
    column = []
    return np.delete(x, column, 1)

def get_training_data(original, valid_set_size):
    if original:
        filename = "train.csv"
        filepath = DIRECTORY + filename
        # TODO
    else:
        x_filename = "train_X"
        y_filename = "train_Y"
        x_filepath = DIRECTORY + x_filename
        y_filepath = DIRECTORY + y_filename

        if os.path.exists(x_filepath) and os.path.exists(y_filepath):
            x = pd.read_csv(x_filepath, dtype=float).as_matrix()
            y = pd.read_csv(y_filepath, header=None).as_matrix().flatten()
            
            if valid_set_size > 0:
                x_train = x[:-valid_set_size]
                y_train = y[:-valid_set_size]
                x_valid = x[-valid_set_size:]
                y_valid = y[-valid_set_size:]
            else:
                x_train = x
                y_train = y
                x_valid = []
                y_valid = []
            
            return x_train, y_train, x_valid, y_valid
        
def get_accuracy(y_hypo, y):
    return len(y_hypo[y_hypo==y]) / len(y_hypo)
    
def get_testing_data(original):
    if original:
        filename = "test.csv"
        # TODO
    else:
        filename = "test_X"
    filepath = DIRECTORY + filename

    if os.path.exists(filepath):
        x = pd.read_csv(filepath, dtype=float).as_matrix()
        return x
        
def output_prediction(y_test, filename="output.csv"):
    arr = [[i+1, int(y_test[i])] for i in range(len(y_test))]
    dw = pd.DataFrame(arr, columns = ["id", "label"])
    dw.to_csv(filename, index=False)
    
def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 1e-10, 1-1e-10)

def normalize(x_set, norm_column=[]):
    x_all = np.concatenate(x_set, axis=0)
    mean = np.mean(x_all, axis=0)
    std = np.std(x_all, axis=0)
    
    for x in x_set:
        for column in norm_column:
            x[:, column] = np.subtract(x[:, column], mean[column])
            x[:, column] = np.true_divide(x[:, column], std[column])
            
    return x_set


# In[ ]:


# Get data
x_train, y_train, x_valid, y_valid = get_training_data(False, 0)
x_test = get_testing_data(False)


# In[ ]:


# Feature scaling (normalize)
if len(x_valid) > 0:
    x_train, x_valid, x_test = normalize([x_train, x_valid, x_test], norm_column)
    x_valid = extract_feature(x_valid)
else:
    x_train, x_test = normalize([x_train, x_test], norm_column)
    
x_train = extract_feature(x_train)
x_test = extract_feature(x_test)


# In[ ]:


x_train = np.concatenate((x_train, x_train**2, x_train**3), axis=1)
x_valid = np.concatenate((x_valid, x_valid**2, x_valid**3), axis=1)
x_test = np.concatenate((x_test, x_test**2, x_test**3), axis=1)


# ### hw2_best

# In[ ]:


# Keras NN model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

y_train_b =  to_categorical(y_train)
y_valid_b = to_categorical(y_valid)

model = Sequential()
model.add(Dense(input_dim=len(x_train[0]), units=50, activation="relu"))
# model.add(Dropout(0.6))
# model.add(Dense(units=50, activation="relu"))
# model.add(Dropout(0.6))

earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

model.add(Dense(units=2, activation="softmax"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train_b, batch_size=100, epochs=40, callbacks=[earlyStopping], validation_data=(x_valid, y_valid_b))

score = model.evaluate(x_valid, y_valid_b)
print('Total loss on Testing Set:', score[0])
print('Accuracy of Testing Set:', score[1])

y_test = model.predict(x_test)
prob = np.argmax(y_test, axis=1)
output_prediction(prob , "best.csv")


# In[ ]:


# Support vector classifier (SVC)
from sklearn.svm import SVC
import pickle

svc = SVC(kernel='rbf')
svc.fit(x_train, y_train)

svc.score(x_train, y_train)

y_test_svc = svc.predict(x_test)
output_prediction(y_test_svc, "svc.csv")


# In[ ]:


# with open('svc.pickle', 'wb') as f:
#     pickle.dump(svc, f)

