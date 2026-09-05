from tensorflow.keras.models import load_model

# Load your model
model = load_model("models/mudra_bilstm.h5")

# Show the architecture
model.summary()
