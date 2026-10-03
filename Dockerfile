# Smart Attendance System: web edition, built for Hugging Face Spaces (Docker SDK).
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    DEEPFACE_HOME=/home/user/.cache \
    SMART_ATTENDANCE_DB_PATH=/home/user/app/data/attendance.db
WORKDIR /home/user/app

COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir --user -r requirements-web.txt

COPY --chown=user . .

# FaceNet weights (88 MB) are fetched at build time instead of being committed,
# then verified against the checksum recorded in MODEL_PROVENANCE.md.
ARG FACENET_URL=https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
ARG FACENET_SHA256=90659cc97bfda5999120f95d8e122f4d262cca11715a21e59ba024bcce816d5c
RUN if [ ! -f models/facenet_weights.h5 ]; then curl -fsSL "$FACENET_URL" -o models/facenet_weights.h5; fi \
 && echo "$FACENET_SHA256  models/facenet_weights.h5" | sha256sum -c -

# Fail the build early if the real model or the API is broken.
RUN python smoke_test_ml.py && python -m unittest discover -s tests

EXPOSE 7860
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:7860", "webapp.app:create_app()"]
