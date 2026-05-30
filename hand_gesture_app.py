from __future__ import annotations

import argparse
import csv
import math
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"
DEFAULT_DATASET_PATH = Path("data") / "gesture_samples.csv"
DEFAULT_CLASSIFIER_PATH = Path("models") / "gesture_classifier.joblib"

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

FINGER_TIPS = (4, 8, 12, 16, 20)
ANGLE_TRIPLES = (
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
)


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float


class GestureSmoother:
    def __init__(self, window_size: int = 9) -> None:
        self.window: deque[str] = deque(maxlen=window_size)

    def update(self, gesture: str) -> str:
        if gesture == "No Hand":
            self.window.clear()
            return gesture

        self.window.append(gesture)
        return Counter(self.window).most_common(1)[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML hand gesture recognition")
    parser.add_argument(
        "--mode",
        choices=("predict", "collect", "train", "summary"),
        default="predict",
        help="Run live prediction, collect samples, train, or summarize labels",
    )
    parser.add_argument("--label", help="Gesture label to collect, for example fist")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MediaPipe hand_landmarker.task model",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="CSV file used for collected gesture samples",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=DEFAULT_CLASSIFIER_PATH,
        help="Trained gesture classifier path",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Minimum hand detection/tracking confidence",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect",
    )
    return parser.parse_args()


def ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmark model to {model_path}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except OSError as exc:
        raise RuntimeError(
            "Model download failed. Check your internet connection or download "
            f"{MODEL_URL} manually and save it as {model_path}."
        ) from exc
    return model_path


def as_points(landmarks: Iterable[object]) -> list[Point]:
    return [Point(lm.x, lm.y, lm.z) for lm in landmarks]


def distance(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    ab = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float32)
    cb = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float32)
    denom = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom == 0:
        return 0.0

    cosine = float(np.dot(ab, cb) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def feature_names() -> list[str]:
    names: list[str] = []
    for idx in range(21):
        names.extend((f"p{idx}_x", f"p{idx}_y", f"p{idx}_z"))
    names.extend(f"tip{tip}_dist" for tip in FINGER_TIPS)
    names.extend(f"angle_{a}_{b}_{c}" for a, b, c in ANGLE_TRIPLES)
    return names


def extract_features(points: list[Point]) -> list[float]:
    wrist = points[0]
    scale = max(distance(points[0], points[9]), distance(points[5], points[17]), 0.001)
    features: list[float] = []

    for point in points:
        features.extend(
            (
                (point.x - wrist.x) / scale,
                (point.y - wrist.y) / scale,
                (point.z - wrist.z) / scale,
            )
        )

    for tip in FINGER_TIPS:
        features.append(distance(points[0], points[tip]) / scale)

    for a, b, c in ANGLE_TRIPLES:
        features.append(angle_degrees(points[a], points[b], points[c]) / 180.0)

    return features


def create_landmarker(args: argparse.Namespace) -> vision.HandLandmarker:
    model_path = ensure_model(args.model)
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max(1, args.max_hands),
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_index)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def detect_points(
    landmarker: vision.HandLandmarker,
    frame,
    timestamp_ms: int,
) -> list[Point] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not results.hand_landmarks:
        return None
    return as_points(results.hand_landmarks[0])


def draw_hand_landmarks(frame, points: list[Point]) -> None:
    height, width = frame.shape[:2]
    pixel_points = [(int(point.x * width), int(point.y * height)) for point in points]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixel_points[start], pixel_points[end], (75, 200, 255), 2)

    for idx, point in enumerate(pixel_points):
        radius = 7 if idx in FINGER_TIPS else 5
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
        cv2.circle(frame, point, radius, (255, 255, 255), 2)


def draw_panel(frame, lines: list[str]) -> None:
    panel_height = 34 + (30 * len(lines))
    cv2.rectangle(frame, (16, 16), (560, panel_height), (18, 18, 18), -1)
    cv2.rectangle(frame, (16, 16), (560, panel_height), (70, 170, 255), 2)

    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (32, 54 + (idx * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def append_sample(dataset_path: Path, label: str, features: list[float]) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not dataset_path.exists()
    with dataset_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["label", *feature_names()])
        writer.writerow([label, *features])


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No dataset found at {dataset_path}")

    labels: list[str] = []
    rows: list[list[float]] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels.append(row["label"])
            rows.append([float(row[name]) for name in feature_names()])

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    return np.array(rows, dtype=np.float32), np.array(labels)


def run_collect(args: argparse.Namespace) -> int:
    if not args.label:
        print("Collect mode needs a label, for example: --mode collect --label fist")
        return 1

    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return 1

    samples = 0
    recording = False
    start_time = time.perf_counter()

    with create_landmarker(args) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            points = detect_points(landmarker, frame, timestamp_ms)

            status = "No hand"
            if points:
                draw_hand_landmarks(frame, points)
                status = "Ready"
                if recording:
                    append_sample(args.dataset, args.label, extract_features(points))
                    samples += 1
                    status = "Recording"

            draw_panel(
                frame,
                [
                    f"Collect: {args.label}",
                    f"Status: {status} | Samples: {samples}",
                    "Space record/pause | q/Esc quit",
                ],
            )
            cv2.imshow("Collect Gesture Samples", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording

    capture.release()
    cv2.destroyAllWindows()
    print(f"Saved {samples} samples for '{args.label}' to {args.dataset}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, labels = load_dataset(args.dataset)
    counts = Counter(labels)
    if len(counts) < 2:
        print("Need at least 2 gesture labels before training.")
        return 1
    if min(counts.values()) < 5:
        print("Each gesture needs at least 5 samples before training.")
        return 1

    stratify = labels if min(counts.values()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    classifier = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        ),
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    args.classifier.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": classifier,
            "feature_names": feature_names(),
            "labels": sorted(counts),
        },
        args.classifier,
    )

    print(f"Trained labels: {', '.join(sorted(counts))}")
    print(f"Samples: {len(labels)}")
    print(f"Accuracy on holdout set: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved classifier to {args.classifier}")
    return 0


def run_summary(args: argparse.Namespace) -> int:
    print(f"Dataset: {args.dataset}")
    if args.dataset.exists():
        _, labels = load_dataset(args.dataset)
        counts = Counter(labels)
        print("Collected labels:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
    else:
        print("Collected labels: none")

    print(f"Classifier: {args.classifier}")
    if args.classifier.exists():
        from joblib import load

        artifact = load(args.classifier)
        labels = artifact.get("labels", [])
        print("Trained labels:")
        for label in labels:
            print(f"  {label}")
    else:
        print("Trained labels: none")

    return 0


def run_predict(args: argparse.Namespace) -> int:
    from joblib import load

    if not args.classifier.exists():
        print(f"No trained classifier found at {args.classifier}")
        print("Collect samples first, then run: .\\run_hand_gesture.bat --mode train")
        return 1

    artifact = load(args.classifier)
    classifier = artifact["model"]

    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return 1

    smoother = GestureSmoother()
    start_time = time.perf_counter()
    fps = 0.0
    ticks = cv2.getTickCount()

    with create_landmarker(args) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            points = detect_points(landmarker, frame, timestamp_ms)

            gesture = "No Hand"
            confidence = 0.0
            if points:
                draw_hand_landmarks(frame, points)
                sample = np.array([extract_features(points)], dtype=np.float32)
                probabilities = classifier.predict_proba(sample)[0]
                best_idx = int(np.argmax(probabilities))
                gesture = str(classifier.classes_[best_idx])
                confidence = float(probabilities[best_idx])

            gesture = smoother.update(gesture)

            now = cv2.getTickCount()
            elapsed = (now - ticks) / cv2.getTickFrequency()
            ticks = now
            if elapsed > 0:
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

            draw_panel(
                frame,
                [
                    f"Gesture: {gesture}",
                    f"Confidence: {confidence:.2f} | FPS: {fps:.1f}",
                    "q/Esc quit",
                ],
            )
            cv2.imshow("ML Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    capture.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()

    if args.mode == "collect":
        return run_collect(args)
    if args.mode == "train":
        return run_train(args)
    if args.mode == "summary":
        return run_summary(args)
    return run_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
