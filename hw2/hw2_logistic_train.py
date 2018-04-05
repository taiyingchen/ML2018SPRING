
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
x_train, y_train, x_valid, y_valid = get_training_data(False, 1000)
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


# ### hw2_logistic

# In[ ]:


w = np.zeros(len(x_train[0]))
b = 0
w_ada = np.zeros(len(x_train[0]))
b_ada = 0
iteration = 5000
lr = 1

x_train_t = x_train.T
for i in range(iteration):
    z = np.dot(x_train, w) + b
    y_train_hypo = sigmoid(z)
    
    w_grad = -np.dot(x_train_t, (y_train - y_train_hypo))
    b_grad = np.sum(-(y_train - y_train_hypo))
    w_ada += w_grad ** 2
    b_ada += b_grad ** 2
    
    w = w - lr / np.sqrt(w_ada) * w_grad
    b = b - lr / np.sqrt(b_ada) * b_grad

    if i % 100 == 0:
        x_entropy = -(np.dot(y_train, np.log(y_train_hypo)) + np.dot((1-y_train), np.log(1-y_train_hypo)))
        print("%i accuracy: %f, cross entropy: %f" % (i, get_accuracy(np.round(y_train_hypo), y_train), x_entropy))


# In[ ]:


z = np.dot(x_valid, w) + b
y_valid_hypo = sigmoid(z)
y_valid_hypo = np.round(y_valid_hypo)
print(get_accuracy(y_valid_hypo, y_valid))


# In[ ]:


np.save("logistic_w", w)
np.save("logistic_b", b)


# In[ ]:


z = np.dot(x_test, w) + b
y_test_hypo = sigmoid(z)
y_test_hypo = np.round(y_test_hypo)
output_prediction(y_test_hypo, "logistic.csv")

