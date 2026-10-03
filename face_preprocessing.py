"""Shared FaceNet input preparation for enrollment and live recognition."""

import cv2
import numpy as np


FACE_SIZE = (160, 160)


def preprocess_face(face_bgr):
    """Convert one OpenCV BGR face crop to a standardized RGB FaceNet input."""
    face = np.asarray(face_bgr)
    if face.ndim != 3 or face.shape[2] != 3 or face.size == 0:
        raise ValueError("FaceNet input must be a non-empty BGR image with three channels.")

    resized_bgr = cv2.resize(face, FACE_SIZE)
    face_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype("float32")
    std = float(face_rgb.std())
    if std == 0.0:
        raise ValueError("FaceNet input has zero variance and cannot be standardized.")
    return (face_rgb - float(face_rgb.mean())) / std
