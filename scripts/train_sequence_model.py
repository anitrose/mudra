import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# --- Paths ---
DATA_FILE = "datasets/preprocessed_sequences.npz"
MODEL_FILE = "models/mudra_bilstm.h5"
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

# --- Load data ---
data = np.load(DATA_FILE)
X = data['X']  # shape: (num_sequences, seq_length, features)
y = data['y']

# --- Encode labels ---
encoder = LabelBinarizer()
y_enc = encoder.fit_transform(y)
num_classes = y_enc.shape[1]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y)

# --- Build Bi-LSTM model ---
seq_length = X.shape[1]
input_dim = X.shape[2]

model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(seq_length, input_dim)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Callbacks ---
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Train ---
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop])

# --- Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# --- Save model ---
model.save(MODEL_FILE)
print(f"Saved Bi-LSTM model to {MODEL_FILE}")

# --- Plot history ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()
