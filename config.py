"""Local configuration and project-relative paths for Smart Attendance System.

Copy ``.env.example`` to ``.env`` and fill in local secrets.  The small loader
below intentionally avoids adding a new runtime dependency just for Phase 1.
Environment variables always take precedence over values in ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file() -> None:
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _path_setting(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


_load_env_file()

MODELS_DIR = _path_setting("SMART_ATTENDANCE_MODELS_DIR", "models")
ANTISPOOFING_MODELS_DIR = _path_setting(
    "SMART_ATTENDANCE_ANTISPOOFING_MODELS_DIR", "antispoofing_models"
)
PHOTOS_DIR = _path_setting("SMART_ATTENDANCE_PHOTOS_DIR", "Photos")
DATASET_DIR = _path_setting("SMART_ATTENDANCE_DATASET_DIR", "dataset")
ATTENDANCE_DETAILS_DIR = _path_setting(
    "SMART_ATTENDANCE_REPORTS_DIR", "Attendance_Details"
)

FACENET_WEIGHTS_PATH = MODELS_DIR / "facenet_weights.h5"
FACE_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
LIVENESS_MODEL_JSON_PATH = ANTISPOOFING_MODELS_DIR / "finalyearproject_antispoofing_model_mobilenet.json"
LIVENESS_MODEL_WEIGHTS_PATH = ANTISPOOFING_MODELS_DIR / "finalyearproject_antispoofing_model_74-0.986316.h5"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.pickle"
RECOGNIZER_PATH = MODELS_DIR / "recognizer.pickle"

DATABASE_CONFIG = {
    "host": os.getenv("SMART_ATTENDANCE_DB_HOST", "localhost"),
    "user": os.getenv("SMART_ATTENDANCE_DB_USER", "root"),
    "password": os.getenv("SMART_ATTENDANCE_DB_PASSWORD", ""),
    "database": os.getenv("SMART_ATTENDANCE_DB_NAME", "recognition"),
}
CAMERA_INDEX = int(os.getenv("SMART_ATTENDANCE_CAMERA_INDEX", "0"))

EMAIL_CONFIG = {
    "host": os.getenv("SMART_ATTENDANCE_EMAIL_HOST", "smtp.gmail.com"),
    "port": int(os.getenv("SMART_ATTENDANCE_EMAIL_PORT", "587")),
    "username": os.getenv("SMART_ATTENDANCE_EMAIL_USERNAME", ""),
    "password": os.getenv("SMART_ATTENDANCE_EMAIL_PASSWORD", ""),
    "sender": os.getenv("SMART_ATTENDANCE_EMAIL_SENDER", ""),
}

REQUIRED_MODEL_FILES = (
    FACENET_WEIGHTS_PATH,
    FACE_CASCADE_PATH,
    LIVENESS_MODEL_JSON_PATH,
    LIVENESS_MODEL_WEIGHTS_PATH,
)
REQUIRED_UI_ASSETS = (
    "new.ico", "Hopstarter-Soft-Scraps-User-Group.ico", "back.png", "adminl.png",
    "profile.png", "password.png", "face.png", "ma.png", "fa.png", "fe.png",
    "tr.png", "Att.png",
)


def missing_startup_assets() -> list[Path]:
    missing = [path for path in REQUIRED_MODEL_FILES if not path.is_file()]
    missing.extend(PHOTOS_DIR / name for name in REQUIRED_UI_ASSETS if not (PHOTOS_DIR / name).is_file())
    return missing
