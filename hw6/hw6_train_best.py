import argparse
import logging
import os

import numpy as np
import pandas as pd
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Add, BatchNormalization, Dense, Dot, Embedding,
                          Flatten, Input)
from keras.models import Model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Set logging config
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument("training_data_path")
# parser.add_argument("movies_path")
# parser.add_argument("users_path")
parser.add_argument("model_path")
# Model arguments
parser.add_argument("--norm_method", default=None, type=str)
parser.add_argument("--test_size", default=0.1, type=float)
parser.add_argument("--latent_dim", default=7, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--cp_monitor", default="val_rmse", type=str)
args = parser.parse_args()


def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.square((y_pred - y_true))))


def get_model(n_users, n_movies):
    logging.info("Get matrix factorization model")
    movie_input = Input(shape=[1], name="Movie")
    movie_embedding = Embedding(
        n_movies, args.latent_dim, name="Movie-Embedding")(movie_input)
    movie_vec = Flatten(name="Flatten-Movie")(movie_embedding)
    movie_bias = Flatten()(Embedding(n_movies, 1, name="Movie-Bias")(movie_input))

    user_input = Input(shape=[1], name="User")
    user_embedding = Embedding(
        n_users, args.latent_dim, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-User")(user_embedding)
    user_bias = Flatten()(Embedding(n_users, 1, name="User-Bias")(user_input))

    movie_bias = Dense(32, activation="linear")(movie_bias)
    movie_bias = Dense(1, activation="linear")(movie_bias)
    user_bias = Dense(32, activation="linear")(user_bias)
    user_bias = Dense(1, activation="linear")(user_bias)

    prob = Dot(axes=-1, name="Dot-Product")([movie_vec, user_vec])
    prob = BatchNormalization()(prob)
    prob = Dense(32, activation="linear")(prob)
    prob = Add(name="Prob")([movie_bias, user_bias, prob])
    prob = Dense(1, activation="linear")(prob)

    model = Model(inputs=[user_input, movie_input], outputs=prob)
    model.compile("adam", loss="mse", metrics=[rmse])
    model.summary()

    return model


def main():
    logging.info("Load training data")
    dataset = pd.read_csv(args.training_data_path, names=[
                          "id", "user_id", "movie_id", "rating"], header=0)
    # logging.info("Load movies data")
    # movies = pd.read_csv(args.movies_path, sep="::", emgine="python")
    # logging.info("Load users data")
    # users = pd.read_csv(args.users_path, sep="::", emgine="python")

    logging.info("Normalization")
    if args.norm_method == "min_max":
        dataset.rating = (dataset.rating - 1.0) / 4.0
    elif args.norm_method == "z_score":
        mu = dataset.rating.mean()
        sigma = dataset.rating.std()
        dataset.rating = preprocessing.scale(dataset.rating)
    else:
        logging.info("Without normalization")

    logging.info("Split to training and validation set")
    train, test = train_test_split(dataset, test_size=args.test_size)
    n_users, n_movies = dataset.user_id.max() + 1, dataset.movie_id.max() + 1
    model = get_model(n_users, n_movies)

    logging.info("Start training")
    ensure_dir(args.model_path)
    checkpoint = ModelCheckpoint(args.model_path,
                                 monitor=args.cp_monitor,
                                 verbose=1,
                                 save_best_only=True)

    history = model.fit([train.user_id, train.movie_id], train.rating,
                        validation_data=(
                            [test.user_id, test.movie_id], test.rating),
                        callbacks=[EarlyStopping(patience=3), checkpoint],
                        batch_size=args.batch_size,
                        epochs=100)


if __name__ == "__main__":
    main()
