from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle
import sys
import os.path

# Parameters
inputfile, outputfile = sys.argv[3], sys.argv[4]
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
        x_filepath = sys.argv[1]
        y_filepath = sys.argv[2]

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

def get_testing_data(original):
    if original:
        filename = "test.csv"
        # TODO
    else:
        filename = "test_X"
    filepath = inputfile

    if os.path.exists(filepath):
        x = pd.read_csv(filepath, dtype=float).as_matrix()
        return x
        
def output_prediction(y_test, filename="output.csv"):
    arr = [[i+1, int(y_test[i])] for i in range(len(y_test))]
    dw = pd.DataFrame(arr, columns = ["id", "label"])
    dw.to_csv(filename, index=False)

def normalize(x_set, norm_column=[]):
    x_all = np.concatenate(x_set, axis=0)
    mean = np.mean(x_all, axis=0)
    std = np.std(x_all, axis=0)
    
    for x in x_set:
        for column in norm_column:
            x[:, column] = np.subtract(x[:, column], mean[column])
            x[:, column] = np.true_divide(x[:, column], std[column])
            
    return x_set

def main():
    x_train, y_train, x_valid, y_valid = get_training_data(False, 0)
    x_test = get_testing_data(False)
    x_train, x_test = normalize([x_train, x_test], norm_column)
    x_test = extract_feature(x_test)
    x_test = np.concatenate((x_test, x_test**2, x_test**3), axis=1)

    with open("model/svc.pickle", "rb") as f:
        svc = pickle.load(f)

    y_test_svc = svc.predict(x_test)
    output_prediction(y_test_svc, outputfile)

if __name__ == "__main__":
    main()
