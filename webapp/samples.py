"""Bundled sample people so the live demo never looks empty.

Every image in ``samples/`` is enrolled at startup through the same pipeline as
a visitor (file name -> person name, e.g. ``Eileen_Collins.jpg``). Sample
people are never auto-cleared, and each day they get one check-in whose
confidence is a real recognition score of their photo.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2

log = logging.getLogger("smart_attendance")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _to_data_url(img) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def seed_samples(db, engine, samples_dir) -> dict[int, float]:
    """Enroll sample photos (idempotent). Returns {student_id: recognition confidence}."""
    folder = Path(samples_dir)
    if not folder.is_dir():
        return {}
    confidences = {}
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT):
        name = path.stem.replace("_", " ").strip()
        img = cv2.imread(str(path))
        if img is None:
            log.warning("sample %s could not be read", path.name)
            continue
        try:
            sid = db.sample_id(name)
            if sid is None:
                sid = db.add_student(name, engine.embed_photos([_to_data_url(img)]), is_sample=True,
                                     created_at="2000-01-01T00:00:00")
            # Score a brightened, slightly blurred copy the way a camera frame would be scored.
            probe = cv2.GaussianBlur(cv2.convertScaleAbs(img, alpha=1.05, beta=12), (3, 3), 0)
            ids, names, gallery = db.gallery()
            faces = engine.faces_in(probe, limit=1)
            if faces:
                match_id, _, score = engine.match(faces[0].embedding, ids, names, gallery)
                if match_id == sid:
                    confidences[sid] = round(score, 4)
            log.info("sample enrolled: %s (id=%s)", name, sid)
        except ValueError as exc:
            log.warning("sample %s skipped: %s", path.name, exc)
    return confidences


def ensure_sample_checkins(db, confidences: dict[int, float]) -> None:
    for sid, conf in confidences.items():
        db.mark_present(sid, conf)  # no-op if already checked in today
