"""Non-GUI smoke test for the local FaceNet and Haar-cascade pipeline."""

import sys

import cv2
import numpy as np
import tensorflow as tf

from config import FACE_CASCADE_PATH, FACENET_WEIGHTS_PATH
from face_preprocessing import FACE_SIZE, preprocess_face
from model_loader import EMBEDDING_DIMENSION, INPUT_SHAPE, load_facenet_model, predict_embedding


def fail(message):
    print("SMOKE TEST FAILED: {}".format(message), file=sys.stderr)
    raise SystemExit(1)


def main():
    print("TensorFlow:", tf.__version__)
    print("OpenCV:", cv2.__version__)

    cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if cascade.empty():
        fail("Haar cascade did not load: {}".format(FACE_CASCADE_PATH))
    print("Haar cascade loaded:", FACE_CASCADE_PATH)

    model = load_facenet_model(FACENET_WEIGHTS_PATH)
    if tuple(model.input_shape[1:]) != INPUT_SHAPE:
        fail("Unexpected model input shape: {}".format(model.input_shape))
    if model.output_shape[-1] != EMBEDDING_DIMENSION:
        fail("Unexpected model output shape: {}".format(model.output_shape))
    print("FaceNet input shape:", model.input_shape)
    print("FaceNet output shape:", model.output_shape)

    # A deterministic synthetic BGR image tests preprocessing and inference
    # without requiring a real person's image or a detected face.
    rows, cols = np.indices(FACE_SIZE)
    synthetic_bgr = np.dstack((rows % 256, cols % 256, (rows + cols) % 256)).astype("uint8")
    prepared = preprocess_face(synthetic_bgr)
    embedding = predict_embedding(model, prepared)

    if embedding.shape != (EMBEDDING_DIMENSION,):
        fail("Unexpected embedding shape: {}".format(embedding.shape))
    if not np.isfinite(embedding).all():
        fail("Embedding contains NaN or infinity.")
    print("Synthetic image preprocessed:", prepared.shape, prepared.dtype)
    print("128-dimensional embedding generated successfully.")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
