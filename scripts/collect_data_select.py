import cv2
import mediapipe as mp
import os
import time
import json
import numpy as np

# --- Config ---
CONFIG_FILE = "mudras.json"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

single_hand = config.get("single_hand", [])
two_hand = config.get("two_hand", [])
RECORD_TIME = config.get("record_time", 10)
COUNTDOWN = config.get("countdown", 5)

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Camera (DroidCam or Webcam) ---
cap = cv2.VideoCapture(0)   # try 1 or 2 for DroidCam


def save_sequence(sequence, label):
    folder = os.path.join(DATA_DIR, label)
    os.makedirs(folder, exist_ok=True)
    filename = f"{int(time.time())}.npy"
    np.save(os.path.join(folder, filename), sequence)

# --- Selection Menu ---
all_mudras = single_hand + two_hand
print("\nAvailable Mudras:")
for i, mudra in enumerate(all_mudras, 1):
    print(f"{i}. {mudra}")

selection = input("\nEnter mudra names (comma-separated) or numbers (e.g., 1,3,5): ").strip()

# Parse selection
selected_mudras = []
if selection:
    if selection.replace(",", "").isdigit():  
        # numbers given
        indices = [int(x.strip()) for x in selection.split(",") if x.strip().isdigit()]
        selected_mudras = [all_mudras[i-1] for i in indices if 0 < i <= len(all_mudras)]
    else:
        # names given
        names = [x.strip() for x in selection.split(",")]
        selected_mudras = [m for m in all_mudras if m in names]

if not selected_mudras:
    print("❌ No valid mudras selected. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# --- Iteration count ---
try:
    iterations = int(input("\nHow many recordings per mudra? (default=1): ").strip() or "1")
except ValueError:
    iterations = 1

print(f"\n✅ Selected mudras: {selected_mudras}")
print(f"✅ Iterations per mudra: {iterations}")

# --- Collect Data ---
for mudra_name in selected_mudras:
    for rep in range(1, iterations+1):
        input(f"\nNext: {mudra_name} (Iteration {rep}/{iterations}). Press ENTER to start countdown...")

        # Countdown
        start_count = time.time()
        while time.time() - start_count < COUNTDOWN:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            remaining = COUNTDOWN - int(time.time() - start_count)
            cv2.putText(frame, f"Get Ready: {mudra_name}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.putText(frame, str(remaining), (250,250),
                        cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255),5)
            cv2.imshow("Mudra Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): exit()

        print(f"Recording {mudra_name} for {RECORD_TIME} seconds... (Iteration {rep}/{iterations})")
        start_time = time.time()
        sequence = []
        while time.time() - start_time < RECORD_TIME:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            frame_data = {'left': None, 'right': None}
            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, hand in zip(result.multi_hand_landmarks, result.multi_handedness):
                    coords = np.array([[pt.x, pt.y, pt.z] for pt in lm.landmark])
                    if hand.classification[0].label.lower() == 'left':
                        frame_data['left'] = coords
                    else:
                        frame_data['right'] = coords
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            sequence.append(frame_data)
            elapsed = int(time.time() - start_time)
            cv2.putText(frame, f"Recording: {mudra_name} {elapsed}/{RECORD_TIME}s",
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.imshow("Mudra Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): exit()
        save_sequence(sequence, mudra_name)
        print(f"✅ Saved sequence for {mudra_name} (Iteration {rep}/{iterations})")

cap.release()
cv2.destroyAllWindows()
print("🎉 Selective mudra data collection complete!")
