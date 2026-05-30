from __future__ import annotations

import argparse
import math
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"

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


FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_PIPS = {
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float


class GestureSmoother:
    def __init__(self, window_size: int = 7) -> None:
        self.window: deque[str] = deque(maxlen=window_size)

    def update(self, gesture: str) -> str:
        if gesture == "No Hand":
            self.window.clear()
            return gesture

        self.window.append(gesture)
        return Counter(self.window).most_common(1)[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live hand gesture recognition")
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
    return math.hypot(a.x - b.x, a.y - b.y)


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    ab_len = math.hypot(*ab)
    cb_len = math.hypot(*cb)
    if ab_len == 0 or cb_len == 0:
        return 0.0

    cosine = ((ab[0] * cb[0]) + (ab[1] * cb[1])) / (ab_len * cb_len)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def palm_size(points: list[Point]) -> float:
    return max(distance(points[0], points[9]), 0.001)


def count_open_fingers(points: list[Point]) -> dict[str, bool]:
    wrist = points[0]
    palm = palm_size(points)
    open_fingers: dict[str, bool] = {}

    # Thumb is harder than other fingers because it moves sideways. Combine
    # straightness, separation from the palm, and distance from the wrist.
    thumb_tip = points[FINGER_TIPS["thumb"]]
    thumb_cmc = points[1]
    thumb_mcp = points[2]
    thumb_ip = points[3]
    open_fingers["thumb"] = (
        angle_degrees(thumb_cmc, thumb_mcp, thumb_ip) > 130
        and angle_degrees(thumb_mcp, thumb_ip, thumb_tip) > 135
        and distance(wrist, thumb_tip) > distance(wrist, thumb_mcp) + (0.35 * palm)
        and distance(thumb_tip, points[5]) > 0.35 * palm
    )

    for finger, (mcp_idx, pip_idx, dip_idx, tip_idx) in FINGER_JOINTS.items():
        mcp = points[mcp_idx]
        pip = points[pip_idx]
        dip = points[dip_idx]
        tip = points[tip_idx]
        pip_angle = angle_degrees(mcp, pip, dip)
        dip_angle = angle_degrees(pip, dip, tip)
        finger_length = distance(mcp, tip)
        first_segment = distance(mcp, pip)
        open_fingers[finger] = (
            pip_angle > 145
            and dip_angle > 150
            and finger_length > first_segment * 1.75
            and distance(wrist, tip) > distance(wrist, pip) + (0.08 * palm)
        )

    return open_fingers


def thumb_direction(points: list[Point]) -> str:
    tip = points[FINGER_TIPS["thumb"]]
    mcp = points[2]
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y

    if abs(dx) > abs(dy):
        return "Thumbs Right" if dx > 0 else "Thumbs Left"
    return "Thumbs Down" if dy > 0 else "Thumbs Up"


def classify_gesture(points: list[Point]) -> tuple[str, dict[str, bool]]:
    open_fingers = count_open_fingers(points)
    opened = {name for name, is_open in open_fingers.items() if is_open}
    palm = palm_size(points)

    thumb_index_close = (
        distance(points[FINGER_TIPS["thumb"]], points[FINGER_TIPS["index"]]) < 0.33 * palm
    )

    if thumb_index_close and {"middle", "ring", "pinky"}.issubset(opened):
        return "OK Sign", open_fingers
    if opened in ({"index", "pinky"}, {"thumb", "index", "pinky"}):
        return "Rock Sign", open_fingers
    if len(opened) == 5:
        return "Open Palm", open_fingers
    if not opened:
        return "Fist", open_fingers
    if opened == {"thumb"}:
        return thumb_direction(points), open_fingers
    if opened == {"index", "middle"}:
        return "Peace Sign", open_fingers
    if opened == {"index"}:
        return "Pointing", open_fingers
    if opened == {"index", "middle", "ring"}:
        return "Three Fingers", open_fingers
    if opened == {"index", "middle", "ring", "pinky"}:
        return "Four Fingers", open_fingers

    return f"{len(opened)} Finger(s)", open_fingers


def draw_status_panel(
    frame,
    gesture: str,
    open_fingers: dict[str, bool],
    fps: float,
    show_help: bool,
) -> None:
    panel_height = 138 if show_help else 88
    cv2.rectangle(frame, (16, 16), (430, panel_height), (18, 18, 18), -1)
    cv2.rectangle(frame, (16, 16), (430, panel_height), (70, 170, 255), 2)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (32, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:0.1f}",
        (32, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (210, 230, 255),
        2,
        cv2.LINE_AA,
    )

    if show_help:
        finger_text = "Open: " + ", ".join(
            finger for finger, is_open in open_fingers.items() if is_open
        )
        if finger_text == "Open: ":
            finger_text = "Open: none"
        cv2.putText(
            frame,
            finger_text,
            (32, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 230, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "q/Esc quit | h help",
            (32, 134),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 230, 255),
            1,
            cv2.LINE_AA,
        )


def draw_hand_landmarks(frame, points: list[Point]) -> None:
    height, width = frame.shape[:2]
    pixel_points = [(int(point.x * width), int(point.y * height)) for point in points]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pixel_points[start], pixel_points[end], (75, 200, 255), 2)

    for idx, point in enumerate(pixel_points):
        radius = 7 if idx in FINGER_TIPS.values() else 5
        cv2.circle(frame, point, radius, (20, 20, 20), -1)
        cv2.circle(frame, point, radius, (255, 255, 255), 2)


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_index)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main() -> int:
    args = parse_args()
    model_path = ensure_model(args.model)
    capture = open_camera(args.camera, args.width, args.height)

    if not capture.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1 or close other camera apps.")
        return 1

    show_help = True
    fps = 0.0
    ticks = cv2.getTickCount()
    start_time = time.perf_counter()
    smoother = GestureSmoother()

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max(1, args.max_hands),
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame could not be read.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = "No Hand"
            open_fingers = {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            }

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    points = as_points(hand_landmarks)
                    gesture, open_fingers = classify_gesture(points)
                    draw_hand_landmarks(frame, points)
                    break

            gesture = smoother.update(gesture)

            now = cv2.getTickCount()
            elapsed = (now - ticks) / cv2.getTickFrequency()
            ticks = now
            if elapsed > 0:
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

            draw_status_panel(frame, gesture, open_fingers, fps, show_help)
            cv2.imshow("Live Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_help = not show_help

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
