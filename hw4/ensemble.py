from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os
import sys

X_TRAIN_PATH = sys.argv[1]
Y_TRAIN_PATH = sys.argv[2]

def get_training_data(valid_set_size):
    x = pd.read_csv(X_TRAIN_PATH, dtype=float).as_matrix()
    y = pd.read_csv(Y_TRAIN_PATH, header=None).as_matrix().flatten()
    
    # Normalize before split into training and validation set
    x = normalize(x)
    # Add power of 2 and 3 terms
    x = np.concatenate((x, x**2, x**3), axis=1)

    x_train = x[:-valid_set_size]
    y_train = y[:-valid_set_size]
    x_valid = x[-valid_set_size:]
    y_valid = y[-valid_set_size:]
    
    return (x_train, y_train), (x_valid, y_valid)

def get_accuracy(y_hypo, y):
    return len(y_hypo[y_hypo==y]) / len(y_hypo)

def normalize(x, norm_column=[0, 10, 78, 79, 80]):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    for column in norm_column:
        x[:, column] = np.subtract(x[:, column], mean[column])
        x[:, column] = np.true_divide(x[:, column], std[column]+1e-10)
    
    return x

def main():
    validation_size = 3000
    (x_train, y_train), (x_valid, y_valid) = get_training_data(validation_size)
    # Decision Tree
    dt_clf = DecisionTreeClassifier(random_state=0)
    dt_clf.fit(x_train, y_train)
    dt_acc = dt_clf.score(x_valid, y_valid)
    
    # Random Forest
    rf_clf = RandomForestClassifier(oob_score=True,random_state=0)
    rf_clf.fit(x_train, y_train)
    rf_acc = rf_clf.score(x_valid, y_valid)

    print("Training size: {}".format(len(x_train)))
    print("Accuracy on validation set (size of {})".format(validation_size))
    print("Decision tree: {:.5f}".format(dt_acc))
    print("Random forest: {:.5f}".format(rf_acc))
    print("Random forest(OOB): {:.5f}".format(rf_clf.oob_score_))

if __name__ == "__main__":
    main()