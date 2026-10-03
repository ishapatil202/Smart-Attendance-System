"""Flask REST API + single-page front end for the Smart Attendance System.

Endpoints
    POST /api/enroll               {name, photo | photos[]}  -> stores face embeddings
    POST /api/recognize            {frame}                   -> matched names + confidence
    GET  /api/attendance[?date=]                             -> attendance records
    GET  /api/attendance/export[?date=]                      -> CSV download
    GET  /api/health                                         -> model status + counts
    GET  /api/metrics                                        -> recognition latency stats
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import statistics
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from .db import Database
from .face_engine import FaceEngine, decode_image
from .samples import seed_samples, ensure_sample_checkins

STATIC_DIR = Path(__file__).resolve().parent / "static"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
log = logging.getLogger("smart_attendance")


class EngineNotReady(Exception):
    def __init__(self, detail=None):
        super().__init__(detail)
        self.detail = detail


def create_app(db_path: str | None = None, engine: FaceEngine | None = None, samples_dir: str | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    db = Database(db_path or os.getenv("SMART_ATTENDANCE_DB_PATH", "data/attendance.db"))
    retention_hours = int(os.getenv("SMART_ATTENDANCE_RETENTION_HOURS", "24"))
    samples_dir = samples_dir or os.getenv("SMART_ATTENDANCE_SAMPLES_DIR", str(Path(__file__).resolve().parent.parent / "samples"))
    state = {"engine": None, "error": None, "load_seconds": None, "samples": {}}
    latencies: deque[dict] = deque(maxlen=1000)

    def on_engine_ready(eng: FaceEngine):
        state["samples"] = seed_samples(db, eng, samples_dir)
        state["engine"] = eng
        ensure_sample_checkins(db, state["samples"])

    if engine is not None:
        on_engine_ready(engine)
    else:
        def _load():
            started = time.perf_counter()
            try:
                on_engine_ready(FaceEngine())
                state["load_seconds"] = round(time.perf_counter() - started, 1)
                log.info("Face engine ready in %.1fs", state["load_seconds"])
            except Exception as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"
                log.exception("Face engine failed to load")
        threading.Thread(target=_load, daemon=True).start()

    def get_engine() -> FaceEngine:
        if state["engine"] is None:
            raise EngineNotReady(state["error"])
        return state["engine"]

    # ---------------------------------------------------------------- errors
    @app.errorhandler(EngineNotReady)
    def _not_ready(exc):
        msg = f"Face model failed to load: {exc.detail}" if exc.detail else "The AI model is still loading. Try again in a few seconds."
        return jsonify(error=msg), 503

    @app.errorhandler(ValueError)
    def _bad_value(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def _too_big(_exc):
        return jsonify(error="Upload too large (8 MB max)."), 413

    # ----------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC_DIR, name)

    # ------------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        eng = state["engine"]
        return jsonify(
            status="ready" if eng else ("error" if state["error"] else "loading"),
            model=eng.embedder.name if eng else None,
            threshold=eng.threshold if eng else None,
            load_seconds=state["load_seconds"],
            error=state["error"],
            retention_hours=retention_hours,
            **db.counts(),
        )

    @app.post("/api/enroll")
    def enroll():
        body = request.get_json(silent=True) or {}
        name = re.sub(r"\s+", " ", str(body.get("name", ""))).strip()
        photos = body.get("photos") or ([body["photo"]] if body.get("photo") else [])
        if not (2 <= len(name) <= 60):
            raise ValueError("Name must be 2 to 60 characters.")
        if not isinstance(photos, list) or not (1 <= len(photos) <= 5):
            raise ValueError("Send 1 to 5 photos.")

        engine = get_engine()
        db.purge_visitors(retention_hours)
        vectors = engine.embed_photos(photos)

        ids, names, gallery = db.gallery()
        for v in vectors:
            dup_id, dup_name, score = engine.match(v, ids, names, gallery)
            if dup_id is not None:
                raise ValueError(f"This face is already enrolled as {dup_name} ({score:.0%} match).")

        sid = db.add_student(name, vectors)
        log.info("enroll student_id=%s embeddings=%d", sid, len(vectors))
        return jsonify(student_id=sid, name=name, embeddings_stored=len(vectors)), 201

    @app.post("/api/recognize")
    def recognize():
        body = request.get_json(silent=True) or {}
        engine = get_engine()
        t0 = time.perf_counter()
        img = decode_image(body.get("frame", ""))
        ids, names, gallery = db.gallery()
        faces = engine.faces_in(img)
        t_model = time.perf_counter()
        results = []
        for face in faces:
            sid, name, score = engine.match(face.embedding, ids, names, gallery)
            item = {"box": face.box, "name": name, "confidence": round(score, 3)}
            if sid is None:
                item["status"] = "unknown"
            else:
                item["status"] = "marked" if db.mark_present(sid, score) else "already_marked"
            results.append(item)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append({"ms": ms, "faces": len(faces), "model_ms": (t_model - t0) * 1000})
        log.info("recognize faces=%d matched=%d gallery=%d latency_ms=%.1f",
                 len(faces), sum(r["name"] is not None for r in results), len(ids), ms)
        h, w = img.shape[:2]
        return jsonify(faces=results, frame={"width": w, "height": h}, latency_ms=round(ms, 1))

    @app.get("/api/attendance")
    def attendance():
        day = _date_arg()
        ensure_sample_checkins(db, state["samples"])
        db.purge_visitors(retention_hours)
        return jsonify(date=day, records=db.attendance(day))

    @app.get("/api/attendance/export")
    def export_csv():
        day = _date_arg()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Name", "Date", "Time", "Confidence", "Type"])
        for r in db.attendance(day, limit=100_000):
            writer.writerow([r["name"], r["date"], r["time"], f"{r['confidence']:.3f}", "sample" if r["is_sample"] else "live"])
        fname = f"attendance_{day or 'all'}.csv"
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})

    @app.get("/api/metrics")
    def metrics():
        with_face = [x["ms"] for x in latencies if x["faces"]]
        def summary(vals):
            if not vals:
                return None
            vals = sorted(vals)
            return {"count": len(vals), "mean_ms": round(statistics.fmean(vals), 1),
                    "p50_ms": round(vals[len(vals) // 2], 1),
                    "p95_ms": round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 1)}
        return jsonify(all_requests=summary([x["ms"] for x in latencies]), requests_with_face=summary(with_face),
                       note="In-memory, since last restart. Each request also logs latency_ms to stdout.")

    def _date_arg():
        day = request.args.get("date") or None
        if day == "today":
            day = date.today().isoformat()
        if day and not DATE_RE.match(day):
            raise ValueError("date must be YYYY-MM-DD")
        return day

    app.extensions["sa_db"] = db
    app.extensions["sa_state"] = state
    return app


# Production: gunicorn "webapp.app:create_app()"
if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")), debug=False)
