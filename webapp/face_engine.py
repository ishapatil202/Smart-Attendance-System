"""Face detection + FaceNet embeddings + open-set matching.

Pipeline (same building blocks as the original desktop app):

    JPEG frame from browser
      -> OpenCV Haar cascade face detection        (models/haarcascade_frontalface_default.xml)
      -> crop + per-face standardisation           (face_preprocessing.preprocess_face)
      -> FaceNet Inception-ResNet-v1, 128-d vector  (model_loader.load_facenet_model)
      -> cosine similarity vs. enrolled templates   (threshold => known / unknown)

The desktop app trained a LinearSVC on the embeddings. A closed-set classifier
always picks *somebody*, which is wrong for a public demo where most faces are
strangers, so the web edition compares against every stored embedding with a
similarity threshold instead (open-set recognition). New students are usable
immediately, with no retraining step.
"""

from __future__ import annotations

import base64
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_preprocessing import preprocess_face  # noqa: E402  (shared with desktop app)

CASCADE_PATH = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"
FACENET_WEIGHTS = PROJECT_ROOT / "models" / "facenet_weights.h5"
MAX_SIDE = 640  # frames are downscaled to this before detection


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]  # x, y, w, h in the (resized) frame
    embedding: np.ndarray           # L2-normalised 128-d


def decode_image(data_url: str) -> np.ndarray:
    """Decode a `data:image/jpeg;base64,...` string (or bare base64) to BGR."""
    if not isinstance(data_url, str) or not data_url:
        raise ValueError("Image payload is missing.")
    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Image is not valid base64.") from exc
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be decoded.")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def l2_normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32").reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class FaceNetEmbedder:
    """The project's FaceNet-128d model (DeepFace Inception-ResNet-v1 + local weights)."""

    name = "FaceNet-128d (Inception-ResNet-v1)"

    def __init__(self, weights_path=FACENET_WEIGHTS):
        from model_loader import load_facenet_model, predict_embedding  # heavy: imports TensorFlow

        self._predict = predict_embedding
        self._model = load_facenet_model(weights_path)
        self._lock = threading.Lock()

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        prepared = preprocess_face(face_bgr)
        with self._lock:
            return l2_normalise(self._predict(self._model, prepared))


class PixelEmbedder:
    """Tiny deterministic stand-in used only by the automated tests (no TensorFlow)."""

    name = "pixel-test-embedder"

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        prepared = preprocess_face(face_bgr)
        small = cv2.resize(cv2.cvtColor(prepared.astype("float32"), cv2.COLOR_RGB2GRAY), (16, 8))
        return l2_normalise(small)


class FaceEngine:
    def __init__(self, embedder=None, threshold: float | None = None):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} has no CascadeClassifier (removed in OpenCV 5). "
                "Install the pinned version: pip install -r requirements-test.txt"
            )
        self.detector = cv2.CascadeClassifier(str(CASCADE_PATH))
        if self.detector.empty():
            raise RuntimeError(f"Haar cascade failed to load from {CASCADE_PATH}")
        if embedder is None:
            kind = os.getenv("SMART_ATTENDANCE_EMBEDDER", "facenet").lower()
            embedder = PixelEmbedder() if kind == "pixel" else FaceNetEmbedder()
        self.embedder = embedder
        # Cosine similarity needed to accept a match (DeepFace's FaceNet cosine
        # distance threshold is 0.40, i.e. similarity 0.60).
        self.threshold = float(threshold if threshold is not None else os.getenv("SMART_ATTENDANCE_MATCH_THRESHOLD", "0.60"))

    # -------------------------------------------------------------- detection
    def detect(self, img: np.ndarray):
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        min_side = max(60, min(img.shape[:2]) // 6)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(min_side, min_side))
        return sorted((tuple(int(v) for v in f) for f in faces), key=lambda b: -b[2] * b[3])

    @staticmethod
    def crop(img, box, margin=0.08):
        x, y, w, h = box
        m = int(margin * w)
        H, W = img.shape[:2]
        return img[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]

    def faces_in(self, img: np.ndarray, limit: int = 5) -> list[DetectedFace]:
        return [DetectedFace(box, self.embedder.embed(self.crop(img, box))) for box in self.detect(img)[:limit]]

    # ------------------------------------------------------------- enrollment
    def embed_photos(self, photos: list[str]) -> list[np.ndarray]:
        """One embedding per photo (largest face). Photos are discarded afterwards."""
        vectors = []
        for data_url in photos:
            img = decode_image(data_url)
            boxes = self.detect(img)
            if boxes:
                vectors.append(self.embedder.embed(self.crop(img, boxes[0])))
        if not vectors:
            raise ValueError("No face found in the photo. Face the camera in good light and try again.")
        return vectors

    # ------------------------------------------------------------ recognition
    def match(self, embedding: np.ndarray, student_ids, names, gallery: np.ndarray):
        """Best match over every stored embedding. Returns (student_id, name, similarity)."""
        if gallery.shape[0] == 0:
            return None, None, 0.0
        sims = gallery @ embedding
        i = int(np.argmax(sims))
        score = float(sims[i])
        if score >= self.threshold:
            return student_ids[i], names[i], score
        return None, None, score
