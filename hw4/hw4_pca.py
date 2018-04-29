# coding: utf-8
import numpy as np
from skimage import io
import os
import sys

IMAGE_FOLDER = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
height = width = 600
channel = 3

def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

def reconstruction(image, eigen_faces, X_mean):
    c = np.dot((image-X_mean), eigen_faces.T)
    image_rec = np.dot(c, eigen_faces)
    image_rec += X_mean
    return image_rec

def output_image(image):
    image = image.reshape(height, width, channel)
    image -= np.min(image)
    image /= np.max(image)
    image = (image*255).astype(np.uint8)
    io.imsave(OUTPUT_PATH, image)

def main():
    image_files = sorted(os.listdir(IMAGE_FOLDER))
    image_paths = [os.path.join(IMAGE_FOLDER, file) for file in image_files]
    X_train = []
    for i in image_paths:
        X_train.append(io.imread(i).flatten())
    X_train = np.array(X_train)

    # Mean of the faces
    X_mean = np.mean(X_train, axis=0)
    U, s, V = np.linalg.svd((X_train-X_mean).T, full_matrices=False)

    eigen_faces = U.T[:4]
    ensure_dir(OUTPUT_PATH)
    index = image_files.index(os.path.basename(OUTPUT_PATH))
    image = X_train[index]

    # Reconstruction
    image_rec = reconstruction(image, eigen_faces, X_mean)
    output_image(image_rec)

if __name__ == "__main__":
   main() 