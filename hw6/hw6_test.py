import pandas as pd
import numpy as np
import argparse
import os
import logging
from keras import backend as K
from keras.models import load_model

# Set logging config
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument("testing_data_path")
parser.add_argument("output_path")
parser.add_argument("model_path")
# Model arguments
parser.add_argument("--norm_method", default=None, type=str)
args = parser.parse_args()


def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.square((y_pred - y_true))))


def main():
    logging.info("Load testing data")
    testset = pd.read_csv(args.testing_data_path, names=[
                          "id", "user_id", "movie_id"], header=0)

    logging.info("Load matrix factorization model")
    model = load_model(args.model_path, custom_objects={"rmse": rmse})

    logging.info("Predict")
    y_hat = model.predict(
        [testset.user_id, testset.movie_id], verbose=True).squeeze()
    if args.norm_method == "min_max":
        y_hat = y_hat * (5.0 - 1.0) + 1.0
    elif args.norm_method == "z_score":
        raise Exception()
        # y_hat = y_hat * sigma + mu
    y_hat = np.clip(y_hat, 1.0, 5.0)

    logging.info("Output prediction to {}".format(args.output_path))
    arr = [[i+1, y_hat[i]] for i in range(len(y_hat))]
    df = pd.DataFrame(arr, columns=["TestDataID", "Rating"])
    ensure_dir(args.output_path)
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
