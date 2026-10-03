# Approved FaceNet-128d provenance

| Field | Value |
| --- | --- |
| Model | DeepFace FaceNet-128d, Inception-ResNet-v1 |
| Architecture source | [DeepFace `Facenet.py`](https://github.com/serengil/deepface/blob/master/deepface/models/facial_recognition/Facenet.py) |
| Weights source | `https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5` |
| Local filename | `models/facenet_weights.h5` |
| SHA-256 | `90659cc97bfda5999120f95d8e122f4d262cca11715a21e59ba024bcce816d5c` |
| Weight size | 87.9 MB (downloaded 2026-09-08) |
| Source licence | DeepFace implementation is [MIT licensed](https://github.com/serengil/deepface/blob/master/LICENSE). Its FaceNet lineage cites [davidsandberg/facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md), also MIT. Preserve upstream attribution and review underlying training-data terms before distribution. |
| TensorFlow target | 2.15.1 on Python 3.11 / macOS arm64 |
| Input | One 160×160×3 RGB `float32` face crop, standardized per face as `(x - mean) / std` |
| Output | One finite, numeric 128-dimensional `float32` embedding |

## Haar cascade provenance

| Field | Value |
| --- | --- |
| Source | [OpenCV 4.x official cascade](https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml) |
| Local filename | `models/haarcascade_frontalface_default.xml` |
| SHA-256 | `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0` |

Only these approved upstream files were downloaded. `embeddings.pickle` and
`recognizer.pickle` remain deliberately absent and must be generated locally.
