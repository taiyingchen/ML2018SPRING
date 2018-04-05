import pandas as pd
import numpy as np
import sys
import os.path

# Parameters
inputfile, outputfile = sys.argv[1], sys.argv[2]
norm_column = [0, 10, 78, 79, 80]

def extract_feature(x):
    column = []
    return np.delete(x, column, 1)
    
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

def get_prediction(x, w, b):
    z = np.dot(x, w) + b
    prob = sigmoid(z)
    prob = np.round(prob)
    prob = 1 - prob
    return prob

def main():
    if len(sys.argv) < 3:
        print("Usage:", sys.argv[0], "<input file> <output file>")
        sys.exit(1)

    x_test = get_testing_data(False)
    [x_test] = normalize([x_test], norm_column)
    x_test = extract_feature(x_test)

    w = np.load("model/generative_w.npy")
    b = np.load("model/generative_b.npy")

    prob = get_prediction(x_test, w, b)
    output_prediction(prob, outputfile)

if __name__ == "__main__":
    main()
