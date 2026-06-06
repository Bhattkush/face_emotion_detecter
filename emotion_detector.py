"""
Face Emotion Detector
=====================
A beginner-friendly AI project that detects faces and emotions
from your webcam in real time using OpenCV and DeepFace.

Author: Your Name
Project: AI Beginner Project #10
"""

import cv2
from deepface import DeepFace
import time

# ── Config ──────────────────────────────────────────────────────────────────
WINDOW_NAME   = "Face Emotion Detector — Press Q to quit"
FONT          = cv2.FONT_HERSHEY_SIMPLEX
ANALYZE_EVERY = 5   # analyze every N frames (lower = slower but more frequent)

# Emotion → color (BGR) mapping
EMOTION_COLORS = {
    "happy":     (0,   200, 100),
    "sad":       (200,  80,  50),
    "angry":     (0,    50, 220),
    "surprise":  (0,   180, 255),
    "fear":      (180,  0,  180),
    "disgust":   (0,   140,  60),
    "neutral":   (180, 180, 180),
}

EMOTION_EMOJI = {
    "happy":    ":)",
    "sad":      ":(",
    "angry":    ">:(",
    "surprise": ":O",
    "fear":     "D:",
    "disgust":  ">_<",
    "neutral":  ":-|",
}

# ── Helper Functions ─────────────────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=10):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img,  (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img,  (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img,  (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img,  (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90,  color, thickness)


def draw_label_box(frame, text, pos, color, scale=0.65, thickness=2):
    """Draw a filled background label."""
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_confidence_bar(frame, emotion_scores, x, y):
    """Draw a mini bar chart of emotion confidences."""
    bar_w, bar_h, gap = 6, 60, 4
    emotions = list(emotion_scores.items())
    emotions.sort(key=lambda e: e[1], reverse=True)

    for i, (emo, score) in enumerate(emotions[:5]):
        filled = int((score / 100.0) * bar_h)
        col    = EMOTION_COLORS.get(emo, (150, 150, 150))
        bx     = x + i * (bar_w + gap)
        cv2.rectangle(frame, (bx, y),           (bx + bar_w, y - bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, y),           (bx + bar_w, y - filled), col,          -1)
        cv2.putText(frame, emo[0].upper(), (bx, y + 14), FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


# ── Main Loop ────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    print("[INFO] Webcam opened. Press Q to quit.")

    frame_count   = 0
    last_results  = []       # cache last DeepFace results
    fps           = 0
    prev_time     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Lost frame — retrying...")
            continue

        frame_count += 1
        h, w = frame.shape[:2]

        # ── FPS counter ──────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ── Run DeepFace every N frames ──────────────────────────────────────
        if frame_count % ANALYZE_EVERY == 0:
            try:
                last_results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
            except Exception as e:
                last_results = []

        # ── Draw results ─────────────────────────────────────────────────────
        for face in last_results:
            region  = face.get("region", {})
            emotion = face.get("dominant_emotion", "neutral").lower()
            scores  = face.get("emotion", {})

            fx = region.get("x", 0)
            fy = region.get("y", 0)
            fw = region.get("w", 0)
            fh = region.get("h", 0)

            color = EMOTION_COLORS.get(emotion, (180, 180, 180))
            emoji = EMOTION_EMOJI.get(emotion, "")

            # Face bounding box
            draw_rounded_rect(frame, (fx, fy), (fx + fw, fy + fh), color, thickness=2)

            # Emotion label
            label = f"{emoji}  {emotion.upper()}"
            draw_label_box(frame, label, (fx, fy - 10), color)

            # Confidence bars (bottom-right of face box)
            if scores:
                draw_confidence_bar(frame, scores, fx + fw - 60, fy + fh - 10)

        # ── HUD overlay ──────────────────────────────────────────────────────
        faces_found = len(last_results)
        hud = f"Faces: {faces_found}   FPS: {fps:.1f}   Press Q to quit"
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(frame, hud, (10, 20), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detector closed. Goodbye!")


if __name__ == "__main__":
    main()
