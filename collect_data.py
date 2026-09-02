import sqlite3
import cv2
import mediapipe as mp

# Use the public MediaPipe API available in the installed package version.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

LABEL = input("Enter the sign label (e.g., A, B, C): ").strip().upper()
SAMPLES_TO_COLLECT = 150

try:
  conn = sqlite3.connect("hand_dataset.db")
  cursor = conn.cursor()
except sqlite3.Error as e:
  print(f"Database error: {e}")
  exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
  print("Error: Cannot open camera")
  exit(1)

collected_count = 0
recording = False

print(f"Recording data for label '{LABEL}'. Press 's' to start/stop, 'q' to quit.")

try:
  while cap.isOpened() and collected_count < SAMPLES_TO_COLLECT:
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      mp_drawing.draw_landmarks(
          frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
      )

      if recording:
        try:
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
        except sqlite3.Error as e:
          print(f"Error inserting data: {e}")
    else:
      if recording:
        cv2.putText(
            frame,
            "No hand detected!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    status_text = (
        "RECORDING - Press 'S' to stop" if recording else "Press 'S' to start"
    )
    status_color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(
        frame,
        status_text,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("s"), ord("S")):
      recording = not recording
      status = "Started" if recording else "Stopped"
      print(f"Recording {status}")
    elif key in (ord("q"), ord("Q")):
      break

finally:
  conn.commit()
  conn.close()
  cap.release()
  cv2.destroyAllWindows()
  print(f"Done recording {LABEL}! Collected {collected_count} samples.")