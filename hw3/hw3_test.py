import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model

def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

def get_testing_data(filepath):
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        x_test = data["feature"].str.split(expand=True).values.reshape(-1, 48, 48, 1).astype('int')
        return x_test
    
def output_prediction(y_test, filename):
    arr = [[i, int(y_test[i])] for i in range(len(y_test))]
    dw = pd.DataFrame(arr, columns = ["id", "label"])
    dw.to_csv(filename, index=False)

def main():
    # Load model and data distribution
    model = load_model("model/model.h5")
    x_test = get_testing_data(sys.argv[1])
    x_test = x_test / 255
    mean, std = np.load("dist.npy")
    x_test = (x_test - mean) / std

    # Predict
    prob = model.predict(x_test)
    y_test = np.argmax(prob, axis=1)

    # Output to file
    ensure_dir(sys.argv[2])
    output_prediction(y_test, sys.argv[2])
    
if __name__ == "__main__":
    main()