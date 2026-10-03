"""Load and validate the approved DeepFace FaceNet-128d embedding model."""

from pathlib import Path

import numpy as np
from deepface.models.facial_recognition.Facenet import InceptionResNetV1


INPUT_SHAPE = (160, 160, 3)
EMBEDDING_DIMENSION = 128
MODEL_NAME = "DeepFace FaceNet-128d (Inception-ResNet-v1)"


def load_facenet_model(weights_path):
    """Construct the approved architecture and load verified local weights."""
    weights = Path(weights_path)
    if not weights.is_file():
        raise FileNotFoundError(
            "FaceNet weights are missing: {}. See MODEL_PROVENANCE.md.".format(weights)
        )
    model = InceptionResNetV1(dimension=EMBEDDING_DIMENSION)
    model.load_weights(str(weights))
    if tuple(model.input_shape[1:]) != INPUT_SHAPE:
        raise ValueError("FaceNet input shape is {}, expected {}.".format(model.input_shape, INPUT_SHAPE))
    if model.output_shape[-1] != EMBEDDING_DIMENSION:
        raise ValueError(
            "FaceNet output dimension is {}, expected {}.".format(
                model.output_shape[-1], EMBEDDING_DIMENSION
            )
        )
    return model


def validate_embedding(embedding):
    """Return one finite, numeric 128-dimensional embedding or raise clearly."""
    values = np.asarray(embedding)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("FaceNet returned a non-numeric embedding.")
    values = values.reshape(-1)
    if values.size != EMBEDDING_DIMENSION:
        raise ValueError(
            "FaceNet returned {} values; expected {}.".format(values.size, EMBEDDING_DIMENSION)
        )
    if not np.isfinite(values).all():
        raise ValueError("FaceNet returned an embedding containing NaN or infinity.")
    return values.astype("float32", copy=False)


def predict_embedding(model, preprocessed_face):
    """Run one preprocessed 160x160 RGB face through FaceNet and validate it."""
    face = np.asarray(preprocessed_face, dtype="float32")
    if face.shape != INPUT_SHAPE:
        raise ValueError("Preprocessed face shape is {}, expected {}.".format(face.shape, INPUT_SHAPE))
    return validate_embedding(model.predict(np.expand_dims(face, axis=0), verbose=0))
