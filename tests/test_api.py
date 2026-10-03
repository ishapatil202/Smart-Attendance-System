"""API route tests.

    python -m unittest discover tests

Uses a small stand-in embedder so the suite runs in seconds without
TensorFlow. The real FaceNet path is checked by smoke_test_ml.py, which runs
during the Docker build.
"""

import base64
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.app import create_app  # noqa: E402
from webapp.face_engine import FaceEngine, PixelEmbedder  # noqa: E402

FACE = cv2.imread(str(ROOT / "tests" / "fixtures" / "face.jpg"))
BLANK = np.full((480, 640, 3), 127, np.uint8)


def data_url(img):
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def variants(img, n=3):
    return [data_url(cv2.convertScaleAbs(img, alpha=1.0, beta=b)) for b in (0, 10, -10)[:n]]


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.empty_samples = Path(self.tmp.name) / "samples"
        self.empty_samples.mkdir()
        self.app = self.make_app(self.empty_samples)
        self.client = self.app.test_client()

    def make_app(self, samples_dir):
        engine = FaceEngine(embedder=PixelEmbedder(), threshold=0.9)
        return create_app(db_path=str(Path(self.tmp.name) / "t.db"), engine=engine, samples_dir=str(samples_dir))

    def tearDown(self):
        self.app.extensions["sa_db"].close()
        self.tmp.cleanup()

    def enroll(self, name="Test Person", photos=None):
        return self.client.post("/api/enroll", json={"name": name, "photos": photos or variants(FACE)})

    # ---------------------------------------------------------------- tests
    def test_health_and_page(self):
        h = self.client.get("/api/health").get_json()
        self.assertEqual(h["status"], "ready")
        self.assertEqual(h["students"], 0)
        page = self.client.get("/")
        self.assertIn(b"Smart Attendance", page.data)
        page.close()

    def test_enroll_stores_embeddings_not_photos(self):
        r = self.enroll()
        self.assertEqual(r.status_code, 201, r.get_json())
        self.assertEqual(r.get_json()["embeddings_stored"], 3)
        db = self.app.extensions["sa_db"]
        counts = db.counts()
        self.assertEqual((counts["students"], counts["embeddings"]), (1, 3))
        # Embeddings are 128 float32s; nothing image-sized is persisted.
        blobs = db._all("SELECT length(vector) AS n FROM embeddings")
        self.assertTrue(all(b["n"] == 128 * 4 for b in blobs))

    def test_single_photo_field_accepted(self):
        r = self.client.post("/api/enroll", json={"name": "One Photo", "photo": data_url(FACE)})
        self.assertEqual(r.status_code, 201)

    def test_recognize_marks_once_per_day(self):
        self.enroll()
        first = self.client.post("/api/recognize", json={"frame": data_url(FACE)}).get_json()
        self.assertEqual(len(first["faces"]), 1)
        face = first["faces"][0]
        self.assertEqual((face["name"], face["status"]), ("Test Person", "marked"))
        self.assertGreater(face["confidence"], 0.9)
        self.assertIn("latency_ms", first)
        again = self.client.post("/api/recognize", json={"frame": data_url(FACE)}).get_json()
        self.assertEqual(again["faces"][0]["status"], "already_marked")
        records = self.client.get("/api/attendance?date=today").get_json()["records"]
        self.assertEqual([r["name"] for r in records], ["Test Person"])

    def test_unknown_face_is_not_marked(self):
        r = self.client.post("/api/recognize", json={"frame": data_url(FACE)}).get_json()
        self.assertEqual(r["faces"][0]["status"], "unknown")
        self.assertIsNone(r["faces"][0]["name"])
        self.assertEqual(self.client.get("/api/attendance").get_json()["records"], [])

    def test_no_face_frame(self):
        r = self.client.post("/api/recognize", json={"frame": data_url(BLANK)}).get_json()
        self.assertEqual(r["faces"], [])

    def test_export_csv(self):
        self.enroll()
        self.client.post("/api/recognize", json={"frame": data_url(FACE)})
        res = self.client.get("/api/attendance/export?date=today")
        self.assertEqual(res.mimetype, "text/csv")
        self.assertIn("attachment", res.headers["Content-Disposition"])
        lines = res.get_data(as_text=True).strip().splitlines()
        self.assertEqual(lines[0], "Name,Date,Time,Confidence,Type")
        self.assertTrue(lines[1].startswith("Test Person,"))
        self.assertTrue(lines[1].endswith(",live"))

    def test_validation_errors(self):
        cases = [
            {"name": "X", "photos": variants(FACE)},          # name too short
            {"name": "Valid Name", "photos": []},             # no photo
            {"name": "Valid Name", "photos": [data_url(FACE)] * 6},  # too many
            {"name": "Valid Name", "photos": [data_url(BLANK)]},     # no face in photo
            {"name": "Valid Name", "photos": ["not-an-image"]},
        ]
        for body in cases:
            with self.subTest(body=str(body)[:60]):
                r = self.client.post("/api/enroll", json=body)
                self.assertEqual(r.status_code, 400)
                self.assertIn("error", r.get_json())
        self.assertEqual(self.client.post("/api/recognize", json={"frame": "nope"}).status_code, 400)
        self.assertEqual(self.client.get("/api/attendance?date=yesterday").status_code, 400)

    def test_duplicate_face_rejected(self):
        self.assertEqual(self.enroll().status_code, 201)
        dup = self.enroll(name="Someone Else")
        self.assertEqual(dup.status_code, 400)
        self.assertIn("already enrolled as Test Person", dup.get_json()["error"])

    def test_visitors_auto_cleared_but_samples_kept(self):
        samples = Path(self.tmp.name) / "with_samples"
        samples.mkdir()
        cv2.imwrite(str(samples / "Sample_Person.jpg"), FACE)
        self.app.extensions["sa_db"].close()
        (Path(self.tmp.name) / "t.db").unlink()
        self.app = self.make_app(samples)
        self.client = self.app.test_client()

        records = self.client.get("/api/attendance?date=today").get_json()["records"]
        self.assertEqual([(r["name"], r["is_sample"]) for r in records], [("Sample Person", 1)])

        db = self.app.extensions["sa_db"]
        db.add_student("Old Visitor", [np.ones(128, dtype="float32") / np.sqrt(128)], created_at="2001-01-01T00:00:00")
        self.assertEqual(db.counts()["students"], 2)
        self.client.get("/api/attendance")  # triggers retention purge
        self.assertEqual(db.counts()["students"], 1)
        self.assertEqual(db.counts()["embeddings"], 1)  # cascade removed visitor embeddings

    def test_metrics_record_latency(self):
        self.client.post("/api/recognize", json={"frame": data_url(FACE)})
        self.client.post("/api/recognize", json={"frame": data_url(BLANK)})
        m = self.client.get("/api/metrics").get_json()
        self.assertEqual(m["all_requests"]["count"], 2)
        self.assertEqual(m["requests_with_face"]["count"], 1)
        self.assertGreater(m["requests_with_face"]["p50_ms"], 0)


if __name__ == "__main__":
    unittest.main()
