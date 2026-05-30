# ML Hand Gesture Recognition

Python webcam app for live hand gesture recognition using OpenCV, MediaPipe,
and scikit-learn.

The app has two clean model tracks:

- Single-hand model: recognizes each hand independently, such as `fist`,
  `open_palm`, `one_finger`, `peace`, or `thumbs_up`.
- Two-hand model: recognizes a combined pose made by both hands together, such
  as `heart_with_two_hands`.

MediaPipe detects hand landmarks. This app turns those landmarks into numeric
features, then trains classifiers from your own webcam samples.

## Run

From this folder:

```powershell
cd C:\Local_Testing\Home\1
.\run_hand_gesture.bat
```

Live prediction will use whichever trained models exist:

- `models/gesture_classifier.joblib` for single-hand predictions.
- `models/two_hand_gesture_classifier.joblib` for paired two-hand predictions.

## Single-Hand Gestures

Collect samples one label at a time:

```powershell
.\run_hand_gesture.bat --mode collect --label open_palm
.\run_hand_gesture.bat --mode collect --label fist
.\run_hand_gesture.bat --mode collect --label one_finger
.\run_hand_gesture.bat --mode collect --label peace
.\run_hand_gesture.bat --mode collect --label three_fingers
.\run_hand_gesture.bat --mode collect --label four_fingers
.\run_hand_gesture.bat --mode collect --label thumbs_up
```

Train the single-hand model:

```powershell
.\run_hand_gesture.bat --mode train
```

Single-hand data is saved to `data/gesture_samples.csv`.

## Two-Hand Gestures

Use paired collection for gestures where both hands form one meaning:

```powershell
.\run_hand_gesture.bat --mode collect-pair --label heart_with_two_hands
.\run_hand_gesture.bat --mode collect-pair --label two_hands_open
```

Train the two-hand model:

```powershell
.\run_hand_gesture.bat --mode train-pair
```

Two-hand data is saved to `data/two_hand_gesture_samples.csv`.

A paired classifier needs at least two labels. For a heart detector, collect
`heart_with_two_hands` plus at least one non-heart paired pose such as
`two_hands_open` or `not_heart`.

## Collection Controls

- Press `Space` to start/pause recording samples.
- Press `q` or `Esc` to quit.
- Move your hand or hands slightly while holding the same gesture.
- Aim for at least `80-150` samples per label.

Single-hand collect mode records one hand. Pair collect mode records only when
exactly two hands are visible.

## Inspect Dataset

Show collected labels and trained labels:

```powershell
.\run_hand_gesture.bat --mode summary
```

## Options

Try a different camera:

```powershell
.\run_hand_gesture.bat --camera 1
```

Limit prediction to one hand:

```powershell
.\run_hand_gesture.bat --max-hands 1
```

Lower detection confidence if the camera misses your hands:

```powershell
.\run_hand_gesture.bat --confidence 0.5
```

## Notes

- Good lighting and keeping hands fully visible improves recognition.
- If you add new labels, retrain the matching model.
- If `models/hand_landmarker.task` is missing, the app will try to download it automatically.
