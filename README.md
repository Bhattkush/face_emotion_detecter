# Face Emotion Detector 

A real-time AI face emotion detector built with Python, OpenCV, and DeepFace.
Detects faces from your webcam and labels emotions: happy, sad, angry, surprised, fearful, disgusted, neutral.

## Setup

```bash
# 1. Clone or download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the detector
python emotion_detector.py
```

## What You'll See
- A live webcam window
- Colored bounding boxes around each face
- Emotion label (e.g. HAPPY :) ) above the face
- Mini confidence bar chart per face
- FPS counter at the top

Press **Q** to quit.

## How It Works
- **OpenCV** opens your webcam and reads frames
- **DeepFace** (built on TensorFlow) analyzes each frame for faces and emotions
- Results are drawn back onto the frame using OpenCV drawing functions

## Emotions Detected
| Emotion   | Color  |
|-----------|--------|
| Happy     | Green  |
| Sad       | Red    |
| Angry     | Blue   |
| Surprise  | Yellow |
| Fear      | Purple |
| Disgust   | Teal   |
| Neutral   | Gray   |

## Troubleshooting
- **No webcam found**: Make sure your camera is connected and not used by another app
- **Slow FPS**: Increase `ANALYZE_EVERY` in the script (default: 5)
- **Install errors**: Make sure you have Python 3.9+ and pip is up to date
