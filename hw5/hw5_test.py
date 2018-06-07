import argparse
import logging
import os
import pickle
import sys

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Set logging config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument("testing_path")
parser.add_argument("output_path")
parser.add_argument("tokenizer_path")
parser.add_argument("rnn_model_path")
# Model parameters
parser.add_argument("--max_sent_length", default=40, type=int)

args = parser.parse_args()

def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)
        
def get_testing_data():
    logging.info("Load testing data")
    X_test = []
    with open(args.testing_path, 'r') as file:
        # Ignore header
        file.readline()
        for line in file:
            no, text = line.strip().split(',', 1)
            X_test.append(text)
    return X_test

def output_file(output):
    logging.info("Output prediction to file")
    with open(args.output_path, 'w') as file:
        file.write("id,label\n")
        file.write('\n'.join(['{},{}'.format(index, label) for index, label in enumerate(output)]))

def get_tokenizer():
    logging.info("Load tokenizer")
    with open(args.tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

def main():
    X_test = get_testing_data()
    tokenizer = get_tokenizer()
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=args.max_sent_length)
    model = load_model(args.rnn_model_path)
    y_test = model.predict(X_test, batch_size=1024, verbose=True)
    y_test = np.around(y_test).astype(np.int32).flatten()
    output_file(y_test)

if __name__ == "__main__":
    main()