import os
import numpy as np
import json

# --- Paths ---
DATA_DIR = "data"
OUTPUT_FILE = "datasets/preprocessed_sequences.npz"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --- Parameters ---
SEQ_LENGTH = 30  # number of frames per sequence
STRIDE = 5       # sliding window step size

# --- Load mudras ---
with open("mudras.json", "r") as f:
    mudras_config = json.load(f)

all_mudras = mudras_config.get("single_hand", []) + mudras_config.get("two_hand", [])
mudra_to_idx = {name: idx for idx, name in enumerate(all_mudras)}

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

# --- Preprocess sequences ---
X, y = [], []

for mudra_name in all_mudras:
    folder = os.path.join(DATA_DIR, mudra_name)
    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        if not file.endswith('.npy'):
            continue
        data = np.load(os.path.join(folder, file), allow_pickle=True)
        sequence = []
        for frame in data:
            left = np.array(frame['left']) if frame['left'] is not None else np.zeros((21,3))
            right = np.array(frame['right']) if frame['right'] is not None else np.zeros((21,3))

            left = normalize_landmarks(left)
            right = normalize_landmarks(right)

            features = list(left.flatten()) + list(right.flatten())
            features += list(compute_interaction_features(left, right))
            sequence.append(features)

        sequence = np.array(sequence)

        for start in range(0, len(sequence) - SEQ_LENGTH + 1, STRIDE):
            X.append(sequence[start:start+SEQ_LENGTH])
            y.append(mudra_to_idx[mudra_name])

X = np.array(X)
y = np.array(y)

print("Sequences shape:", X.shape, "Labels shape:", y.shape)
np.savez(OUTPUT_FILE, X=X, y=y)
print(f"Saved preprocessed sequences to {OUTPUT_FILE}")