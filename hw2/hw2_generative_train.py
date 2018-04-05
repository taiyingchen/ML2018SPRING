
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


# ### hw2_generative

# In[ ]:


def get_gaussian(x, y):
    mu = [None] * 2
    cov = [None] * 2
    var = [None] * 2
    # Calculate mean
    mu[0] = np.mean(x[y==0], axis=0)
    mu[1] = np.mean(x[y==1], axis=0)

    # Calculate covariance
    cov[0] = np.dot((x[y==0] - mu[0]).T, x[y==0] - mu[0])
    cov[1] = np.dot((x[y==1] - mu[1]).T, x[y==1] - mu[1])
    cov[0] /= len(x[y==0])
    cov[1] /= len(x[y==1])

    return mu, cov

def get_generative_model(x, y):
    mu, cov = get_gaussian(x, y)
    shared_cov = (len(y[y==0]) * cov[0] + len(y[y==1]) * cov[1]) / len(y)
    # Naive bayes
    # shared_cov = np.diag(np.diag(shared_cov))
    
    w = np.dot(mu[0] - mu[1], np.linalg.pinv(shared_cov))
    b = ((-0.5) * np.dot(np.dot(mu[0].T, np.linalg.pinv(shared_cov)), mu[0]) +
        (0.5) * np.dot(np.dot(mu[1].T, np.linalg.pinv(shared_cov)), mu[1]) +
        np.log(len(y[y==0])/len(y[y==1])))
    
    return w, b

def get_prediction(x, w, b):
    z = np.dot(x, w) + b
    prob = sigmoid(z)
    prob = np.round(prob)
    prob = 1 - prob
    return prob


# In[ ]:


w, b = get_generative_model(x_train, y_train)
prob = get_prediction(x_train, w, b)
print("accuracy: %f" % get_accuracy(prob, y_train))


# In[ ]:


prob = get_prediction(x_test, w, b)
output_prediction(prob , "generative.csv")

