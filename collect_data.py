import sqlite3
import cv2
import mediapipe as mp

# Setup Vision & MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

LABEL = input("Enter the sign label (e.g., A, B, C): ").strip().upper()
SAMPLES_TO_COLLECT = 150

# Connect to the initialized database
conn = sqlite3.connect("hand_dataset.db")
cursor = conn.cursor()

cap = cv2.VideoCapture(0)
collected_count = 0
recording = False

print(f"Recording data for label '{LABEL}'. Press 's' to start.")

while cap.isOpened() and collected_count < SAMPLES_TO_COLLECT:
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.flip(frame, 1)
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(rgb_frame)

  if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
    mp.solutions.drawing_utils.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
    )

    if recording:
      row = []
      for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
      row.append(LABEL)

      placeholders = ", ".join(["?"] * 64)
      cursor.execute(
          f"INSERT INTO landmarks VALUES (NULL, {placeholders})", tuple(row)
      )

      collected_count += 1
      cv2.putText(
          frame,
          f"Recording {LABEL}: {collected_count}/{SAMPLES_TO_COLLECT}",
          (20, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0, 0, 255),
          2,
      )

  if not recording:
    cv2.putText(
        frame,
        "Press 'S' to Start",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

  cv2.imshow("Data Collector", frame)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("s"):
    recording = True
  elif key == ord("q"):
    break

conn.commit()
conn.close()
cap.release()
cv2.destroyAllWindows()
print(f"Done recording {LABEL}!")