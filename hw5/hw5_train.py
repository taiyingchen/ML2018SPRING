import argparse
import itertools
import logging
import os
import pickle
import string
import sys

import numpy as np
from gensim.models import word2vec
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import GRU, LSTM, Dense, Dropout, Embedding, Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Set logging config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument("action", choices=["train", "semi"])
parser.add_argument("label_training_path")
parser.add_argument("unlabel_training_path")
parser.add_argument("tokenizer_path")
parser.add_argument("word2vec_model_path")
parser.add_argument("rnn_model_path")
# Model parameters
parser.add_argument("--remove_stopwords", default=False, type=bool)
parser.add_argument("--remove_duplichar", default=False, type=bool)
parser.add_argument("--embed_dim", default=128, type=int)
parser.add_argument("--dict_size", default=20000, type=int)
parser.add_argument("--max_sent_length", default=40, type=int)
parser.add_argument("--threshold", default=0.1, type=float)
parser.add_argument("--val_frac", default=0.1, type=float)
args = parser.parse_args()

def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

def get_label_training_data():
    logging.info("Load label training data")
    X_train = []
    y_train = []
    
    with open(args.label_training_path, 'r') as file:
        for line in file:
            label, text = line.strip().split(" +++$+++ ")
            text = remove_punctuation(text)
            X_train.append(text)
            y_train.append(int(label))
    if args.remove_stopwords:
        X_train = remove_stopwords(X_train)
    if args.remove_duplichar:
        X_train = remove_duplichar(X_train)
    return X_train, y_train

def get_unlabel_training_data():
    logging.info("Load unlabel training data")
    X_train = []

    with open(args.unlabel_training_path, 'r') as file:
        for line in file:
            text = line.strip()
            text = remove_punctuation(text)
            X_train.append(text)
    if args.remove_stopwords:
        X_train = remove_stopwords(X_train)
    if args.remove_duplichar:
        X_train = remove_duplichar(X_train)
    return X_train

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(X):
    """
    Remove stopwords and split sentence to words
    """
    stoplist = set()
    X_ = [[word for word in sentence.split() if word not in stoplist] for sentence in X]
    return X_

def remove_duplichar(X):
    """
    Remove duplicate characters
    """
    for i, text in enumerate(X):
        X[i] = [''.join(ch for ch, _ in itertools.groupby(word)) for word in text]
    return X
        
def word2sent(X):
    for i, v in enumerate(X):
        X[i] = ' '.join(v)
    return X

def sent2word(X):
    for i in range(len(X)):
        X[i] = [word for word in X[i].split()]
    return X

def get_tokenizer(X):
    logging.info("Get tokenizer")
    tokenizer = Tokenizer(num_words=args.dict_size)
    tokenizer.fit_on_texts(X)
    # Save tokenizer
    ensure_dir(args.tokenizer_path)
    with open(args.tokenizer_path, 'wb') as file:
        pickle.dump(tokenizer, file)
    return tokenizer

def to_sequence(X, tokenizer):
    _ = tokenizer.texts_to_sequences(X)
    return pad_sequences(_, maxlen=args.max_sent_length)

def get_word2vec(X):
    logging.info("Get word2vec model")
    X = sent2word(X)
    w2v_model = word2vec.Word2Vec(X, size=args.embed_dim)
    ensure_dir(args.word2vec_model_path)
    w2v_model.save(args.word2vec_model_path)
    return w2v_model
    
def get_embedding_matrix(tokenizer, w2v_model):
    logging.info("Get embedding matrix")
    word_index = tokenizer.word_index
    embeddings_index = {}
    for k, v in w2v_model.wv.vocab.items():
        embeddings_index[k] = w2v_model.wv[k]

    embedding_matrix = np.zeros((args.dict_size + 1, args.embed_dim))
    for word, i in word_index.items():
        if i > args.dict_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_RNN(embedding_matrix):
    logging.info("Get RNN model")
    sequence_input = Input(shape=(args.max_sent_length,), dtype='int32')
    # Embedding
    embedded_sequences = Embedding(embedding_matrix.shape[0],
                                   embedding_matrix.shape[1],
                                   weights=[embedding_matrix],
                                   trainable=False)(sequence_input)
    # RNN
    output = GRU(512,
                return_sequences=True,
                dropout=0.3)(embedded_sequences)
    output = GRU(512,
                return_sequences=True,
                dropout=0.3)(output)
    output = GRU(512,
                return_sequences=False,
                dropout=0.3)(output)
    # DNN
    output = Dense(256,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.1))(output)
    output = Dropout(0.3)(output)
    preds = Dense(1, activation='sigmoid')(output)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    print(model.summary())
    return model

def split_val(X, y):
    val_size = int(len(X) * args.val_frac)
    return (X[:-val_size], y[:-val_size]), (X[-val_size:], y[-val_size:])

def main():
    X_label, y_label = get_label_training_data()
    X_unlabel = get_unlabel_training_data()
    X_all = X_label + X_unlabel
    
    tokenizer = get_tokenizer(X_all)
    X_label = to_sequence(X_label, tokenizer)
    X_unlabel = to_sequence(X_unlabel, tokenizer)
    (X, y), (X_val, y_val) = split_val(X_label, y_label)

    w2v_model = get_word2vec(X_all)
    embedding_matrix = get_embedding_matrix(tokenizer, w2v_model)
    model = get_RNN(embedding_matrix)
    ensure_dir(args.rnn_model_path)
    checkpoint = ModelCheckpoint(args.rnn_model_path,
                                 monitor="val_acc",
                                 verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 mode="max")

    if args.action == "train":
        history = model.fit(X, y, validation_data=(X_val, y_val),
                            epochs=10,
                            batch_size=128,
                            callbacks=[checkpoint])
    elif args.action == "semi":
        raise Exception()

if __name__ == "__main__":
    main()
