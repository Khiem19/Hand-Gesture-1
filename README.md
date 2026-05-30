# Live Hand Gesture Recognition

Python webcam app for live hand gesture recognition using OpenCV and MediaPipe.

This folder includes a local `.venv` and the MediaPipe hand landmark model at
`models/hand_landmarker.task` if setup was completed by Codex.

## Gestures recognized

- Open palm
- Fist
- Thumbs up / down / left / right
- Peace sign
- Pointing index
- Three fingers
- Four fingers
- OK sign
- Rock sign

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `python` does not work on your machine, try `py` instead:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Fastest on Windows:

```powershell
.\run_hand_gesture.bat
```

Or run the Python file directly:

```powershell
python hand_gesture_app.py
```

Optional camera index:

```powershell
python hand_gesture_app.py --camera 1
```

## Controls

- Press `q` or `Esc` to quit.
- Press `h` to toggle the help overlay.

## Notes

- Good lighting and keeping your hand fully visible improves recognition.
- The classifier is tuned for one clear hand facing the camera. Use `--max-hands 2` only if you need two-hand detection.
- If the camera does not open, close other apps that may be using it, then try `--camera 1`.
- If `models/hand_landmarker.task` is missing, the app will try to download it automatically.
