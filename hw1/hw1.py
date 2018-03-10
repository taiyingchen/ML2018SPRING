import pandas as pd
import numpy as np
import sys

FEATURE_PER_DAY = 18
WINDOW_SIZE = 9
FEATURE_NUM = 18 * 9

def main():
    if len(sys.argv) < 3:
        print("Usage:", sys.argv[0], "<input file> <output file>")
        sys.exit(1)
    
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    df = pd.read_csv(inputfile, header=None, encoding="big5")
    for row in np.arange(10, len(df), FEATURE_PER_DAY):
        df.iloc[row, 2:] = pd.to_numeric(df.iloc[row, 2:], errors="coerce")
    df.fillna(0, inplace=True)

    theta = np.load("theta.npy")

    x_test = []
    y_test = []

    for i in np.arange(0, len(df), FEATURE_PER_DAY):
        arr = np.array(df.iloc[i:i+FEATURE_PER_DAY, 2:2+WINDOW_SIZE], dtype=float)
        arr = np.insert(arr, 0, 1)  # x0 for bias
        x_test.append(arr)

    for i in range(len(x_test)):
        y_test.append(np.dot(theta, x_test[i]))

    # Replace negative value
    for i in range(len(y_test)):
        if (y_test[i] < 0):
            y_test[i] = 0

    arr = [["id_" + str(i), y_test[i]] for i in range(len(y_test))]
    dw = pd.DataFrame(arr, columns=["id", "value"])
    dw.to_csv(outputfile, index=False)

if __name__ == "__main__":
    main()