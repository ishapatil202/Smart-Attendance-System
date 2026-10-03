"""Measure recognition accuracy and speed of the deployed pipeline.

Uses exactly the same FaceEngine (detector + FaceNet + cosine threshold) as the
web app, so the numbers can go straight onto a resume.

Usage (inside the Docker image, which has TensorFlow):

    # Labeled Faces in the Wild, downloaded by scikit-learn (~200 MB)
    python scripts/evaluate.py --lfw

    # Or your own folder: eval_data/<person name>/<image>.jpg
    python scripts/evaluate.py --folder eval_data

Protocol: for every person with enough images, the first --enroll images are
enrolled and the rest are used as probes. A probe is correct when the top
match is the right person and clears the threshold. People with only one
image are used as "unknown" probes to measure false accepts.
"""

from __future__ import annotations

import argparse
import base64
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from webapp.face_engine import FaceEngine  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def to_data_url(img_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load_folder(root: Path):
    people = defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                img = cv2.imread(str(f))
                if img is not None:
                    people[d.name.replace("_", " ")].append(img)
    return people


def load_lfw(min_faces: int):
    from sklearn.datasets import fetch_lfw_people

    # Full 250x250 images (slice_=None) so the app's own detector does the cropping.
    lfw = fetch_lfw_people(min_faces_per_person=1, color=True, resize=1.0, slice_=None)
    people = defaultdict(list)
    for img, label in zip(lfw.images, lfw.target):
        people[lfw.target_names[label]].append(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR))
    known = {k: v for k, v in people.items() if len(v) >= min_faces}
    unknown = {k: v for k, v in people.items() if len(v) == 1}
    return known, unknown


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lfw", action="store_true", help="use Labeled Faces in the Wild")
    src.add_argument("--folder", type=Path, help="folder of <person>/<images>")
    ap.add_argument("--enroll", type=int, default=3, help="images per person to enroll")
    ap.add_argument("--min-faces", type=int, default=10, help="LFW: min images per known person")
    ap.add_argument("--max-people", type=int, default=100)
    ap.add_argument("--max-probes", type=int, default=10, help="probe images per person")
    ap.add_argument("--unknown", type=int, default=200, help="LFW: number of unknown-person probes")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    if args.lfw:
        known, unknown_people = load_lfw(args.min_faces)
    else:
        known, unknown_people = load_folder(args.folder), {}
    known = dict(list(sorted(known.items(), key=lambda kv: -len(kv[1])))[: args.max_people])

    engine = FaceEngine(threshold=args.threshold)
    print(f"Model: {engine.embedder.name} | threshold {engine.threshold}")

    # ---- enroll
    ids, names, vectors = [], [], []
    for pid, (name, imgs) in enumerate(known.items()):
        if len(imgs) <= args.enroll:
            continue
        try:
            for v in engine.embed_photos([to_data_url(i) for i in imgs[: args.enroll]]):
                ids.append(pid); names.append(name); vectors.append(v)
        except ValueError:
            pass
    gallery = np.stack(vectors)
    enrolled = sorted(set(names))
    print(f"Enrolled {len(enrolled)} people, {len(vectors)} embeddings")

    # ---- probes: same path as POST /api/recognize (decode -> detect -> embed -> match)
    latencies, correct, wrong, rejected, no_face = [], 0, 0, 0, 0
    for name in enrolled:
        for img in known[name][args.enroll: args.enroll + args.max_probes]:
            t0 = time.perf_counter()
            faces = engine.faces_in(cv2.imdecode(np.frombuffer(cv2.imencode(".jpg", img)[1], np.uint8), 1), limit=1)
            if not faces:
                no_face += 1
                continue
            _, match_name, _ = engine.match(faces[0].embedding, ids, names, gallery)
            latencies.append((time.perf_counter() - t0) * 1000)
            if match_name == name:
                correct += 1
            elif match_name is None:
                rejected += 1
            else:
                wrong += 1

    false_accepts = unknown_total = 0
    for imgs in list(unknown_people.values())[: args.unknown]:
        faces = engine.faces_in(imgs[0], limit=1)
        if faces:
            unknown_total += 1
            if engine.match(faces[0].embedding, ids, names, gallery)[1] is not None:
                false_accepts += 1

    probes = correct + wrong + rejected
    print("\n=== Results ===")
    print(f"Probe images with a detected face: {probes} (no face detected in {no_face})")
    print(f"Top-1 identification accuracy:     {100 * correct / max(probes, 1):.1f}%  ({correct}/{probes})")
    print(f"  wrong person:                    {wrong}")
    print(f"  rejected as unknown:             {rejected}")
    if unknown_total:
        print(f"False-accept rate (strangers):     {100 * false_accepts / unknown_total:.1f}%  ({false_accepts}/{unknown_total})")
    if latencies:
        lat = sorted(latencies)
        print(f"Latency per frame (CPU):           median {statistics.median(lat):.0f} ms, "
              f"p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} ms")


if __name__ == "__main__":
    main()
