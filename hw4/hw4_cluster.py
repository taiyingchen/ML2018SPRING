from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import os
import sys
import pickle

height = width = 28
IMAGE_PATH = sys.argv[1]
TEST_CASE_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
KMEANS_PATH = "kmeans.pickle"
CC_PATH = "cluster_center.npy"

def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    # Load training data
    X_train = np.load(IMAGE_PATH)

    # Load test case
    with open(TEST_CASE_PATH, 'r') as file:
        content = file.read().strip('\n').replace(',', ' ').split()[3:]
        X_test = []
        for i in range(0, len(content), 3):
            X_test.append((content[i+1], content[i+2]))
            
        X_test = np.array(X_test).astype("int")
    
    return X_train, X_test

def dimension_reduction(X_train):
    # Normalization
    X_train_norm = X_train / 255

    pca = PCA(n_components=395, whiten=True)
    X_train_pca = pca.fit_transform(X_train_norm)

    return X_train_pca

def clustering(X_train):
    # with open(KMEANS_PATH, 'rb') as file:
    #     kmeans = pickle.load(file)
    cluster_center = np.load(CC_PATH)
    kmeans = KMeans(n_clusters=2, random_state=0, init=cluster_center, n_init=1).fit(X_train)
    X_train_labels = kmeans.predict(X_train)

    return X_train_labels

def output_file(X_train_labels, X_test):
    # Output to file
    output = []
    cnt = 0
    for i, j in X_test:
        if X_train_labels[i] == X_train_labels[j]:
            cnt += 1
            output.append(1)
        else:
            output.append(0)
    print("Number of same clusters: %i" % cnt)
    with open(OUTPUT_PATH, 'w') as file:
        file.write("ID,Ans\n")
        file.write('\n'.join(['{},{}'.format(index, element) for index, element in enumerate(output)]))

def main():
    X_train, X_test = load_data()
    X_train_pca = dimension_reduction(X_train)
    X_train_labels = clustering(X_train_pca)
    ensure_dir(OUTPUT_PATH)
    output_file(X_train_labels, X_test)

if __name__ == "__main__":
    main()