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
DEFAULT_PAIR_DATASET_PATH = Path("data") / "two_hand_gesture_samples.csv"
DEFAULT_PAIR_CLASSIFIER_PATH = Path("models") / "two_hand_gesture_classifier.joblib"

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


@dataclass(frozen=True)
class HandDetection:
    points: list[Point]
    side: str


@dataclass(frozen=True)
class MotionSignal:
    side: str
    speed: float
    direction: str
    pinch: float
    spread: float
    swipe: str | None


class GestureSmoother:
    def __init__(self, window_size: int = 9) -> None:
        self.window: deque[str] = deque(maxlen=window_size)

    def update(self, gesture: str) -> str:
        if gesture == "No Hand":
            self.window.clear()
            return gesture

        self.window.append(gesture)
        return Counter(self.window).most_common(1)[0][0]


class MotionTracker:
    def __init__(self, history_seconds: float = 0.75, swipe_cooldown: float = 0.75) -> None:
        self.history_seconds = history_seconds
        self.swipe_cooldown = swipe_cooldown
        self.history: dict[str, deque[tuple[float, Point]]] = {}
        self.last_swipe_at: dict[str, float] = {}

    def update(self, side: str, points: list[Point], timestamp_s: float) -> MotionSignal:
        center = hand_center(points)
        history = self.history.setdefault(side, deque())
        history.append((timestamp_s, center))

        while history and timestamp_s - history[0][0] > self.history_seconds:
            history.popleft()

        speed = 0.0
        direction = "Still"
        swipe = None

        if len(history) >= 2:
            old_time, old_center = history[0]
            elapsed = max(timestamp_s - old_time, 0.001)
            dx = center.x - old_center.x
            dy = center.y - old_center.y
            movement = math.hypot(dx, dy)
            speed = movement / elapsed
            direction = movement_direction(dx, dy)

            previous_swipe = self.last_swipe_at.get(side, -self.swipe_cooldown)
            can_swipe = timestamp_s - previous_swipe > self.swipe_cooldown
            if can_swipe and direction != "Still" and movement > 0.20 and speed > 0.45:
                swipe = f"Swipe {direction}"
                self.last_swipe_at[side] = timestamp_s

        scale = hand_scale(points)
        pinch = distance(points[4], points[8]) / scale
        spread = distance(points[8], points[20]) / scale
        return MotionSignal(side, speed, direction, pinch, spread, swipe)

    def clear_missing(self, seen_sides: set[str]) -> None:
        for side in list(self.history):
            if side not in seen_sides:
                del self.history[side]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML hand gesture recognition")
    parser.add_argument(
        "--mode",
        choices=("predict", "collect", "train", "collect-pair", "train-pair", "summary"),
        default="predict",
        help="Run prediction, collect/train single-hand or two-hand samples, or summarize labels",
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
        "--pair-dataset",
        type=Path,
        default=DEFAULT_PAIR_DATASET_PATH,
        help="CSV file used for collected two-hand gesture samples",
    )
    parser.add_argument(
        "--pair-classifier",
        type=Path,
        default=DEFAULT_PAIR_CLASSIFIER_PATH,
        help="Trained two-hand gesture classifier path",
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
        default=2,
        help="Maximum number of hands to detect",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show hand motion/debug details during live prediction",
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


def hand_center(points: list[Point]) -> Point:
    count = len(points)
    return Point(
        sum(point.x for point in points) / count,
        sum(point.y for point in points) / count,
        sum(point.z for point in points) / count,
    )


def hand_scale(points: list[Point]) -> float:
    return max(distance(points[0], points[9]), distance(points[5], points[17]), 0.001)


def movement_direction(dx: float, dy: float) -> str:
    if math.hypot(dx, dy) < 0.035:
        return "Still"
    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"
    return "Down" if dy > 0 else "Up"


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
    scale = hand_scale(points)
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


def pair_feature_names() -> list[str]:
    names: list[str] = []
    for prefix in ("left", "right"):
        names.extend(f"{prefix}_{name}" for name in feature_names())
    names.extend(
        (
            "wrist_dx",
            "wrist_dy",
            "wrist_dz",
            "wrist_distance",
            "index_tip_distance",
            "thumb_tip_distance",
            "pinky_tip_distance",
        )
    )
    return names


def extract_pair_features(hands: list[HandDetection]) -> list[float]:
    if len(hands) != 2:
        raise ValueError("Two-hand features require exactly 2 detected hands.")

    left, right = sorted(hands, key=lambda hand: hand.points[0].x)
    features = [
        *extract_features(left.points),
        *extract_features(right.points),
    ]

    wrist_distance = max(distance(left.points[0], right.points[0]), 0.001)
    features.extend(
        (
            (right.points[0].x - left.points[0].x) / wrist_distance,
            (right.points[0].y - left.points[0].y) / wrist_distance,
            (right.points[0].z - left.points[0].z) / wrist_distance,
            wrist_distance,
            distance(left.points[8], right.points[8]) / wrist_distance,
            distance(left.points[4], right.points[4]) / wrist_distance,
            distance(left.points[20], right.points[20]) / wrist_distance,
        )
    )
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


def detect_hands(
    landmarker: vision.HandLandmarker,
    frame,
    timestamp_ms: int,
) -> list[HandDetection]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not results.hand_landmarks:
        return []

    hands: list[HandDetection] = []
    for landmarks in results.hand_landmarks:
        points = as_points(landmarks)
        side = "Left" if points[0].x < 0.5 else "Right"
        hands.append(HandDetection(points=points, side=side))

    return sorted(hands, key=lambda hand: hand.points[0].x)


def detect_points(
    landmarker: vision.HandLandmarker,
    frame,
    timestamp_ms: int,
) -> list[Point] | None:
    hands = detect_hands(landmarker, frame, timestamp_ms)
    if not hands:
        return None
    return hands[0].points


def draw_hand_landmarks(frame, points: list[Point]) -> None:
    height, width = frame.shape[:2]
    pixel_points = [(int(point.x * width), int(point.y * height)) for point in points]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixel_points[start], pixel_points[end], (75, 200, 255), 2)

    for idx, point in enumerate(pixel_points):
        radius = 7 if idx in FINGER_TIPS else 5
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
        cv2.circle(frame, point, radius, (255, 255, 255), 2)


def draw_hand_label(frame, points: list[Point], text: str) -> None:
    height, width = frame.shape[:2]
    min_x = max(8, int(min(point.x for point in points) * width))
    min_y = max(32, int(min(point.y for point in points) * height) - 14)

    cv2.putText(
        frame,
        text,
        (min_x, min_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (min_x, min_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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


def append_sample(
    dataset_path: Path,
    label: str,
    features: list[float],
    names: list[str] | None = None,
) -> None:
    names = names or feature_names()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not dataset_path.exists()
    with dataset_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["label", *names])
        writer.writerow([label, *features])


def load_dataset(
    dataset_path: Path,
    names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    names = names or feature_names()
    if not dataset_path.exists():
        raise FileNotFoundError(f"No dataset found at {dataset_path}")

    labels: list[str] = []
    rows: list[list[float]] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels.append(row["label"])
            rows.append([float(row[name]) for name in names])

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
    collector_args = argparse.Namespace(**vars(args))
    collector_args.max_hands = 1

    with create_landmarker(collector_args) as landmarker:
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


def train_classifier(
    dataset_path: Path,
    classifier_path: Path,
    names: list[str],
    title: str,
) -> int:
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, labels = load_dataset(dataset_path, names)
    counts = Counter(labels)
    if len(counts) < 2:
        print(f"Need at least 2 {title} labels before training.")
        return 1
    if min(counts.values()) < 5:
        print(f"Each {title} label needs at least 5 samples before training.")
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

    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": classifier,
            "feature_names": names,
            "labels": sorted(counts),
            "type": title,
        },
        classifier_path,
    )

    print(f"Trained labels: {', '.join(sorted(counts))}")
    print(f"Samples: {len(labels)}")
    print(f"Accuracy on holdout set: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved classifier to {classifier_path}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    return train_classifier(
        dataset_path=args.dataset,
        classifier_path=args.classifier,
        names=feature_names(),
        title="single-hand gesture",
    )


def run_train_pair(args: argparse.Namespace) -> int:
    return train_classifier(
        dataset_path=args.pair_dataset,
        classifier_path=args.pair_classifier,
        names=pair_feature_names(),
        title="two-hand gesture",
    )


def run_collect_pair(args: argparse.Namespace) -> int:
    if not args.label:
        print("Collect-pair mode needs a label, for example: --mode collect-pair --label heart")
        return 1

    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return 1

    samples = 0
    recording = False
    start_time = time.perf_counter()
    collector_args = argparse.Namespace(**vars(args))
    collector_args.max_hands = 2

    with create_landmarker(collector_args) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            hands = detect_hands(landmarker, frame, timestamp_ms)

            for hand in hands:
                draw_hand_landmarks(frame, hand.points)

            status = f"Need 2 hands, found {len(hands)}"
            if len(hands) == 2:
                status = "Ready"
                if recording:
                    append_sample(
                        args.pair_dataset,
                        args.label,
                        extract_pair_features(hands),
                        pair_feature_names(),
                    )
                    samples += 1
                    status = "Recording"

            draw_panel(
                frame,
                [
                    f"Collect pair: {args.label}",
                    f"Status: {status} | Samples: {samples}",
                    "Space record/pause | q/Esc quit",
                ],
            )
            cv2.imshow("Collect Two-Hand Gesture Samples", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording

    capture.release()
    cv2.destroyAllWindows()
    print(f"Saved {samples} two-hand samples for '{args.label}' to {args.pair_dataset}")
    return 0


def run_summary(args: argparse.Namespace) -> int:
    print(f"Single-hand dataset: {args.dataset}")
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

    print()
    print(f"Two-hand dataset: {args.pair_dataset}")
    if args.pair_dataset.exists():
        _, labels = load_dataset(args.pair_dataset, pair_feature_names())
        counts = Counter(labels)
        print("Collected two-hand labels:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
    else:
        print("Collected two-hand labels: none")

    print(f"Two-hand classifier: {args.pair_classifier}")
    if args.pair_classifier.exists():
        from joblib import load

        artifact = load(args.pair_classifier)
        labels = artifact.get("labels", [])
        print("Trained two-hand labels:")
        for label in labels:
            print(f"  {label}")
    else:
        print("Trained two-hand labels: none")

    return 0


def run_predict(args: argparse.Namespace) -> int:
    from joblib import load

    single_classifier = None
    pair_classifier = None
    if args.classifier.exists():
        single_classifier = load(args.classifier)["model"]
    if args.pair_classifier.exists():
        pair_classifier = load(args.pair_classifier)["model"]

    if single_classifier is None and pair_classifier is None and not args.details:
        print(f"No trained classifier found at {args.classifier}")
        print(f"No trained two-hand classifier found at {args.pair_classifier}")
        print("Collect samples first, then train a single-hand or two-hand classifier.")
        print("Or run with --details to inspect hand motion without a trained classifier.")
        return 1

    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return 1

    smoothers = {
        "Left": GestureSmoother(),
        "Right": GestureSmoother(),
    }
    pair_smoother = GestureSmoother()
    motion_tracker = MotionTracker()
    show_details = args.details
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
            hands = detect_hands(landmarker, frame, timestamp_ms)
            timestamp_s = timestamp_ms / 1000.0

            predictions: list[tuple[str, str, float]] = []
            motion_signals: list[MotionSignal] = []
            seen_sides: set[str] = set()
            for idx, hand in enumerate(hands, start=1):
                side = hand.side
                if side in seen_sides:
                    side = f"Hand {idx}"
                    smoothers.setdefault(side, GestureSmoother())
                seen_sides.add(side)

                draw_hand_landmarks(frame, hand.points)
                signal = motion_tracker.update(side, hand.points, timestamp_s)
                motion_signals.append(signal)
                if single_classifier is not None:
                    sample = np.array([extract_features(hand.points)], dtype=np.float32)
                    probabilities = single_classifier.predict_proba(sample)[0]
                    best_idx = int(np.argmax(probabilities))
                    gesture = str(single_classifier.classes_[best_idx])
                    confidence = float(probabilities[best_idx])
                    gesture = smoothers[side].update(gesture)
                    predictions.append((side, gesture, confidence))
                    draw_hand_label(frame, hand.points, f"{side}: {gesture}")

            for side, smoother in smoothers.items():
                if side not in seen_sides:
                    smoother.update("No Hand")
            motion_tracker.clear_missing(seen_sides)

            pair_prediction: tuple[str, float] | None = None
            if pair_classifier is not None and len(hands) == 2:
                sample = np.array([extract_pair_features(hands)], dtype=np.float32)
                probabilities = pair_classifier.predict_proba(sample)[0]
                best_idx = int(np.argmax(probabilities))
                gesture = str(pair_classifier.classes_[best_idx])
                confidence = float(probabilities[best_idx])
                pair_prediction = (pair_smoother.update(gesture), confidence)
            else:
                pair_smoother.update("No Hand")

            now = cv2.getTickCount()
            elapsed = (now - ticks) / cv2.getTickFrequency()
            ticks = now
            if elapsed > 0:
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

            if predictions:
                lines = [
                    *(f"{side}: {gesture} ({confidence:.2f})" for side, gesture, confidence in predictions),
                ]
            elif hands:
                lines = [f"Hands detected: {len(hands)}"]
            else:
                lines = ["No Hand"]

            if pair_prediction:
                pair_gesture, pair_confidence = pair_prediction
                lines.append(f"Two-hand: {pair_gesture} ({pair_confidence:.2f})")
            elif pair_classifier is not None:
                lines.append("Two-hand: need exactly 2 hands")

            swipe_events = [signal.swipe for signal in motion_signals if signal.swipe]
            if swipe_events:
                lines.append(f"Motion: {', '.join(swipe_events)}")

            if show_details:
                for signal in motion_signals:
                    lines.append(
                        f"{signal.side} detail: {signal.direction} "
                        f"speed {signal.speed:.2f} pinch {signal.pinch:.2f}"
                    )
                if len(hands) == 2:
                    left, right = sorted(hands, key=lambda hand: hand.points[0].x)
                    avg_scale = (hand_scale(left.points) + hand_scale(right.points)) / 2.0
                    wrist_gap = distance(left.points[0], right.points[0]) / max(avg_scale, 0.001)
                    index_gap = distance(left.points[8], right.points[8]) / max(avg_scale, 0.001)
                    lines.append(f"Pair detail: wrists {wrist_gap:.2f} index tips {index_gap:.2f}")

            lines.extend((f"Hands: {len(hands)} | FPS: {fps:.1f}", "d details | q/Esc quit"))

            draw_panel(frame, lines)
            cv2.imshow("ML Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_details = not show_details

    capture.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()

    if args.mode == "collect":
        return run_collect(args)
    if args.mode == "train":
        return run_train(args)
    if args.mode == "collect-pair":
        return run_collect_pair(args)
    if args.mode == "train-pair":
        return run_train_pair(args)
    if args.mode == "summary":
        return run_summary(args)
    return run_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
