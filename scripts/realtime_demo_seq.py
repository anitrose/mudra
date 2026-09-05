import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

# --- Paths ---
MODEL_FILE = "models/mudra_bilstm.h5"
CONFIG_FILE = "mudras.json"

# --- Load model ---
model = tf.keras.models.load_model(MODEL_FILE)

# --- Load mudra labels ---
with open(CONFIG_FILE, "r") as f:
    mudras_config = json.load(f)
all_mudras = mudras_config.get("single_hand", []) + mudras_config.get("two_hand", [])
idx_to_mudra = {idx:name for idx,name in enumerate(all_mudras)}

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Parameters ---
SEQ_LENGTH = 30
FEATURE_DIM = model.input_shape[2]
SLIDING_WINDOW = []

# --- Helper functions ---
def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks

def compute_interaction_features(left, right):
    features = []
    fingertips = [4, 8, 12, 16, 20]
    for i in fingertips:
        for j in fingertips:
            dist = np.linalg.norm(left[i] - right[j])
            features.append(dist)
    left_palm = np.mean(left[:5], axis=0)
    right_palm = np.mean(right[:5], axis=0)
    features.append(np.linalg.norm(left_palm - right_palm))
    min_dist = np.min([np.linalg.norm(l - r) for l in left for r in right])
    features.append(min_dist)
    touching = 1.0 if min_dist < 0.05 else 0.0
    features.append(touching)
    return np.array(features)

# --- Video capture ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left_row = np.zeros((21,3))
    right_row = np.zeros((21,3))
    hand_detected = False

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_detected = True
        for lm, hand in zip(result.multi_hand_landmarks, result.multi_handedness):
            coords = np.array([[pt.x, pt.y, pt.z] for pt in lm.landmark])
            coords = normalize_landmarks(coords)
            if hand.classification[0].label.lower() == 'left':
                left_row = coords
            else:
                right_row = coords
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    if hand_detected:
        features = list(left_row.flatten()) + list(right_row.flatten()) + list(compute_interaction_features(left_row, right_row))
        if len(features) < FEATURE_DIM:
            features += [0.0]*(FEATURE_DIM - len(features))
        SLIDING_WINDOW.append(features)
        if len(SLIDING_WINDOW) > SEQ_LENGTH:
            SLIDING_WINDOW.pop(0)

        if len(SLIDING_WINDOW) == SEQ_LENGTH:
            X_input = np.array(SLIDING_WINDOW).reshape(1, SEQ_LENGTH, FEATURE_DIM)
            pred_idx = np.argmax(model.predict(X_input, verbose=0), axis=1)[0]
            prediction = idx_to_mudra[pred_idx]
        else:
            prediction = "Detecting..."
    else:
        prediction = "No hand"
        SLIDING_WINDOW = []  # reset sliding window when no hand detected

    cv2.putText(frame, prediction, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.imshow("Mudra Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
