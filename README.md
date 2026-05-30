# ML Hand Gesture Recognition

Python webcam app for live hand gesture recognition using OpenCV, MediaPipe,
and a classifier trained from your own webcam samples.

This folder includes a local `.venv` and the MediaPipe hand landmark model at
`models/hand_landmarker.task` if setup was completed by Codex.

## How it works

MediaPipe detects 21 hand landmarks. This app converts those landmarks into
normalized numeric features, then trains a scikit-learn classifier to recognize
the gesture labels you record.

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

## Collect training samples

Record each gesture separately. Hold the gesture in front of the camera, press
`Space` to start recording, move your hand slightly while keeping the same
gesture, then press `Space` again to pause.

Examples:

```powershell
.\run_hand_gesture.bat --mode collect --label open_palm
.\run_hand_gesture.bat --mode collect --label fist
.\run_hand_gesture.bat --mode collect --label one_finger
.\run_hand_gesture.bat --mode collect --label peace
.\run_hand_gesture.bat --mode collect --label three_fingers
.\run_hand_gesture.bat --mode collect --label four_fingers
.\run_hand_gesture.bat --mode collect --label thumbs_up
```

Aim for at least 80-150 samples per gesture. More variation usually helps:
slightly different distance, position, and angle.

## Train

```powershell
.\run_hand_gesture.bat --mode train
```

This reads `data/gesture_samples.csv` and saves the classifier to
`models/gesture_classifier.joblib`.

To see what labels you collected and what labels the trained classifier knows:

```powershell
.\run_hand_gesture.bat --mode summary
```

## Run live prediction

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
- In collect mode, press `Space` to start/pause recording samples.

## Notes

- Good lighting and keeping your hand fully visible improves recognition.
- The trained classifier is personalized to the gestures you collect.
- If you want `one_finger`, `three_fingers`, or `four_fingers`, collect samples for those labels and retrain.
- Use one clear hand facing the camera. Use `--max-hands 2` only if you need two-hand detection.
- If the camera does not open, close other apps that may be using it, then try `--camera 1`.
- If `models/hand_landmarker.task` is missing, the app will try to download it automatically.
