
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random


# In[ ]:


# Constant
FEATURE_PER_DAY = 18
WINDOW_SIZE = 9
FEATURE_NUM = 18 * 9
DAY_PER_MONTH = 20
VALIDATION_SET = False
SQUARE_TERM = False
REGULARIZATION = False
FEATURE_SCALING = False
if SQUARE_TERM:
    FEATURE_NUM *= 2


# In[ ]:


# Verify model by compute its error
def rmse(x_data, y_data, theta):
    cost = (np.dot(x_data, theta) - y_data) ** 2
    cost = np.sum(cost) / len(x_data)
    return np.sqrt(cost)

# Feature scaling
def feature_scaling(x_data):
    x_data = np.subtract(x_data, np.mean(x_data, axis=0))
    x_data = np.divide(x_data, np.std(x_data, axis=0))
    x_data = np.nan_to_num(x_data)
    return x_data

# Validate training data
def validation(data):
    for element in data:
        if float(element) <= 0:
            return False
    return True


# In[ ]:


# Read training data
df = pd.read_csv("ml-2018spring-hw1/train.csv", encoding="big5")

# Transform RAINFALL column to number
for row in range(10, len(df), FEATURE_PER_DAY):
    df.iloc[row, 3:] = pd.to_numeric(df.iloc[row, 3:], errors="coerce")

df.fillna(0, inplace=True)


# In[ ]:


# Data preprocessing
data = []

for i in range(FEATURE_PER_DAY):
    data.append([])

for index, row in df.iterrows():
    for item in range(3, 27):
        data[index%FEATURE_PER_DAY].append(row[item])
        
data = np.array(data)


# In[ ]:


# Training set
x_data = []
y_data = []
# Validation set
x_v_data = []
y_v_data = []

# Select specific features
sieve = [i for i in range(18)]
FEATURE_NUM = len(sieve) * WINDOW_SIZE

for i in range(len(data[0])):
    if i % 480 + WINDOW_SIZE < 480 and validation(data[9][i:i+WINDOW_SIZE+1]):
        vec = []
        for j in sieve:
            for element in data[j][i:i+WINDOW_SIZE]:
                vec.append(float(element))
        
        if VALIDATION_SET and random.randint(1, 10) % 10 == 0:
            x_v_data.append(vec)
            y_v_data.append(float(data[9][i+WINDOW_SIZE]))
        else:  
            x_data.append(vec)
            y_data.append(float(data[9][i+WINDOW_SIZE]))
    
x_data = np.array(x_data)
y_data = np.array(y_data)

if SQUARE_TERM:
    x_data = np.concatenate((x_data, x_data**2), axis=1)
x_data = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)

if VALIDATION_SET:
    x_v_data = np.array(x_v_data)
    y_v_data = np.array(y_v_data)
    if SQUARE_TERM:
        x_v_data = np.concatenate((x_v_data, x_v_data**2), axis=1)
    x_v_data = np.concatenate((np.ones((x_v_data.shape[0], 1)), x_v_data), axis=1)

if FEATURE_SCALING:
    x_data[:, 1:] = feature_scaling(x_data[:, 1:])
    if VALIDATION_SET:
        x_v_data[:, 1:] = feature_scaling(x_v_data[:, 1:])


# In[ ]:


# Closed-form solution
theta_c = np.linalg.lstsq(x_data, y_data)
theta_c = np.array(theta_c[0])


# In[ ]:


# Initial model
theta = np.array([0.0] * (FEATURE_NUM + 1)) # all parameters init to 0
lr_ada = np.zeros(FEATURE_NUM + 1)
lr = 10
ld = 0.0001
iteration = 15000


# In[ ]:


# Training
x_data_t = x_data.transpose()

for i in range(iteration):
    dot = np.dot(x_data, theta)
    loss = y_data - dot
    grad = np.dot(x_data_t, loss) * (-2.0)
    lr_ada += grad ** 2
    theta = theta - lr / np.sqrt(lr_ada) * grad
    if REGULARIZATION:
        # Don't need to consider bias term
        theta[1:] = (1 - lr * ld) * theta[1:]


# In[ ]:


# Save model to npy file
np.save("theta", theta)


# In[ ]:


# Compute test value
dt = pd.read_csv("ml-2018spring-hw1/test.csv", header=None, encoding="big5")

for row in range(10, len(dt), FEATURE_PER_DAY):
    dt.iloc[row, 2:] = pd.to_numeric(dt.iloc[row, 2:], errors="coerce")
    
dt.fillna(0, inplace=True)

x_test = []
y_test = []
TEST_ROW_SIZE, TEST_COLUMN_SIZE = dt.shape

for i in range(0, len(dt), FEATURE_PER_DAY):
    vec = []
    for j in sieve:
        for element in dt.iloc[j+i][2:]:
            vec.append(float(element))
    if SQUARE_TERM:
        arr = np.concatenate((arr, arr**2)) # Add x^2 terms
    vec = np.insert(vec, 0, 1) # x0 for bias
    x_test.append(vec)
x_test = np.array(x_test)
    
if FEATURE_SCALING:
    x_test[:, 1:] = feature_scaling(x_test[:, 1:])
    
for i in range(len(x_test)):
    y_test.append(np.dot(theta, x_test[i]))

# Replace negative value
for i in range(len(y_test)):
    if y_test[i] < 0:
        y_test[i] = 0

arr = [["id_" + str(i), y_test[i]] for i in range(len(y_test))]
dw = pd.DataFrame(arr, columns = ["id", "value"])
dw.to_csv("output.csv", index=False)


# In[ ]:


print("Training set:")
print("gd: %f" % rmse(x_data, y_data, theta))
print("cf: %f" % rmse(x_data, y_data, theta_c))
if VALIDATION_SET:
    print("Validation set:")
    print("gd: %f" % rmse(x_v_data, y_v_data, theta))
    print("cf: %f" % rmse(x_v_data, y_v_data, theta_c))

