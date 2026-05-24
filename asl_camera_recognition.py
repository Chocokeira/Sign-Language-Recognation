import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
from collections import deque

# Configuration
MODEL = 'asl_final.h5'
CLASSES = 'classes_letters.json'
PRED_BUF = 7  # Buffer size for smoothing predictions

# Load trained model and letter mappings
model = load_model(MODEL)

with open(CLASSES, 'r') as f:
    letters = json.load(f)

# Set up MediaPipe (NEW API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6
)

hands = HandLandmarker.create_from_options(options)

# Buffer for temporal smoothing
buf = deque(maxlen=PRED_BUF)

# Open webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    timestamp = int(time.time() * 1000)

    res = hands.detect_for_video(
    mp_image,
    timestamp
)

    if res.hand_landmarks:

        lm = res.hand_landmarks[0]

        # Extract bounding box around hand
        xcoords = [p.x for p in lm]
        ycoords = [p.y for p in lm]

        x1 = int(min(xcoords) * w) - 30
        x2 = int(max(xcoords) * w) + 30
        y1 = int(min(ycoords) * h) - 30
        y2 = int(max(ycoords) * h) + 30

        # Ensure boundaries are within frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Extract and process hand region
        roi = frame[y1:y2, x1:x2]

        if roi.size:

            # Convert to grayscale and resize to 28x28
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(gray, (28, 28))

            img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0

            # Get prediction
            probs = model.predict(img, verbose=0)[0]

            idx = int(np.argmax(probs))

            conf = float(probs[idx])

            # Add to buffer
            buf.append(idx)

            # Use majority vote when buffer is full
            if len(buf) == buf.maxlen:

                best = max(set(buf), key=buf.count)

                letter = letters[best]

                conf2 = max(probs)

                # Draw results
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{letter} {conf2*100:.0f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    # Show webcam
    cv2.imshow('ASL Recognition', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
