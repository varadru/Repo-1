"""Detect the pose of several people in every frame of a video.

MediaPipe's own multi-pose mode (num_poses > 1) misses people very often, so a
top-down approach is used instead:

1. an object detector finds every "person" box in the frame;
2. the boxes where people were in the previous frame are added too, which keeps
   a dancer alive when the detector misses them (e.g. while partly hidden);
3. each box is cropped and the single-person Pose Landmarker runs on the crop;
4. poses that landed on the same person are merged.
"""
import os
import urllib.request

import cv2
import numpy as np

MODEL_URLS = {
    "pose_lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "detector": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
}
MODEL_FILES = {
    "pose_lite": "pose_landmarker_lite.task",
    "pose_full": "pose_landmarker_full.task",
    "pose_heavy": "pose_landmarker_heavy.task",
    "detector": "efficientdet_lite2.tflite",
}

TORSO = [11, 12, 23, 24]


def ensure_model(key, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, MODEL_FILES[key])
    if not os.path.exists(path):
        print(f"Baixando modelo {MODEL_FILES[key]}...")
        urllib.request.urlretrieve(MODEL_URLS[key], path + ".part")
        os.replace(path + ".part", path)
    return path


def torso_histogram(frame_bgr, image_px):
    """HSV colour histogram of the torso region, used to tell people apart."""
    h, w = frame_bgr.shape[:2]
    pts = image_px[TORSO, :2]
    x0, y0 = np.clip(pts.min(axis=0).astype(int), 0, [w - 1, h - 1])
    x1, y1 = np.clip(pts.max(axis=0).astype(int), 0, [w - 1, h - 1])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([crop], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    total = hist.sum()
    return hist / total if total > 0 else None


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir o vídeo: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, count


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _pose_box(image_px):
    xy = image_px[:, :2]
    return np.concatenate([xy.min(axis=0), xy.max(axis=0)])


def _expand(box, width, height, margin=0.25):
    """Grow a box by `margin` on every side and make it roughly square."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    size = max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * margin)
    x0, y0 = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
    x1, y1 = int(min(width, cx + size / 2)), int(min(height, cy + size / 2))
    return x0, y0, x1, y1


def _torso_center(image_px):
    return image_px[TORSO, :2].mean(axis=0)


class MultiPoseDetector:
    def __init__(self, pose_model, detector_model, max_people, det_conf=0.5, pose_conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        self.max_people = max_people
        self.detector = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=detector_model),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=det_conf * 0.6,
            category_allowlist=["person"],
            max_results=max_people + 3,
        ))
        self.landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=pose_conf,
            min_pose_presence_confidence=pose_conf,
        ))
        self.prev_boxes = []

    def close(self):
        self.detector.close()
        self.landmarker.close()

    def _image(self, rgb):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def __call__(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes = []
        for d in self.detector.detect(self._image(rgb)).detections:
            b = d.bounding_box
            boxes.append((d.categories[0].score, np.array(
                [b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height], dtype=float)))
        boxes.sort(key=lambda sb: -sb[0])
        candidates = [b for _, b in boxes]
        for pb in self.prev_boxes:
            if all(_iou(pb, b) < 0.5 for b in candidates):
                candidates.append(pb)

        dets = []
        for box in candidates:
            x0, y0, x1, y1 = _expand(box, width, height)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            result = self.landmarker.detect(self._image(rgb[y0:y1, x0:x1]))
            if not result.pose_landmarks:
                continue
            lms, wlms = result.pose_landmarks[0], result.pose_world_landmarks[0]
            cw, ch = x1 - x0, y1 - y0
            image_px = np.array([[x0 + l.x * cw, y0 + l.y * ch, l.z * cw] for l in lms])
            vis = np.array([l.visibility if l.visibility is not None else 1.0 for l in lms])
            # Skip a pose that duplicates a person already found from another box.
            size = max(np.ptp(image_px[:, 1]), 1.0)
            if any(np.linalg.norm(_torso_center(image_px) - _torso_center(o["image_px"])) < 0.15 * size
                   for o in dets):
                continue
            dets.append({
                "world": np.array([[l.x, l.y, l.z] for l in wlms]),
                "image_px": image_px,
                "vis": vis,
                "score": float(vis[TORSO].mean()),
                "hist": torso_histogram(frame_bgr, image_px),
            })
            if len(dets) >= self.max_people:
                break

        self.prev_boxes = [_pose_box(d["image_px"]) for d in dets]
        return dets


def extract(video_path, models_dir, pose_variant, max_people, det_conf=0.5, start=0.0, end=None, progress=True):
    """Returns (detections_per_frame, timestamps_s, (fps, width, height)).

    Each detection is a dict with "world" (33x3, MediaPipe world metres),
    "image_px" (33x3 pixel coords, z is relative depth), "vis" (33,) and "hist".
    """
    detector = MultiPoseDetector(
        ensure_model("pose_" + pose_variant, models_dir), ensure_model("detector", models_dir),
        max_people, det_conf=det_conf, pose_conf=det_conf,
    )
    fps, width, height, count = video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    frames, times = [], []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            index += 1
            if t < start:
                continue
            if end is not None and t > end:
                break
            dets = detector(frame)
            frames.append(dets)
            times.append(t)
            if progress and len(frames) % 15 == 0:
                total = f"/{count}" if count > 0 else ""
                print(f"\r  quadro {index}{total}  pessoas: {len(dets)}   ", end="", flush=True)
    finally:
        cap.release()
        detector.close()
    if progress:
        print()
    return frames, np.array(times), (fps, width, height)
