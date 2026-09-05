# AdvancedMudraPredict

A Python-based system for capturing, preprocessing, training, and recognizing Indian dance mudras, including single-hand and two-hand mudras like Anjaly, using MediaPipe and a Bi-LSTM sequence model.

## Repo Structure

```
AdvancedMudraPredict/
├── data/                        # Raw collected sequences per mudra
├── datasets/                    # Preprocessed sequences
│   └── preprocessed_sequences.npz
├── models/                      # Trained models
│   └── mudra_bilstm.h5
├── scripts/                     # Source scripts
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── train_sequence_model.py
│   └── realtime_demo_seq.py
├── mudras.json                  # Mudra labels and config
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Mudras Configuration

Edit `mudras.json` to include your mudras:

```json
{
  "single_hand": ["pathaka_left", "pathaka_right"],
  "two_hand": ["anjaly_both", "katakamukha_both"],
  "record_time": 10,
  "countdown": 5
}
```

## Step 1: Collect Data

Run the script to capture sequences using your webcam:

```bash
python scripts/collect_data.py
```

* Follow the countdown prompts.
* Both hands are captured; frames are saved in `data/<mudra_name>/`.

## Step 2: Preprocess Data

Generate normalized sequences with interaction features:

```bash
python scripts/preprocess.py
```

* Creates `datasets/preprocessed_sequences.npz`.
* Computes fingertip distances, palm distance, min distance, and touching flag.

## Step 3: Train Bi-LSTM Model

Train the sequence model:

```bash
python scripts/train_sequence_model.py
```

* Model saved as `models/mudra_bilstm.h5`.
* Accuracy and loss plots will be displayed.

## Step 4: Real-Time Mudra Recognition

Run real-time recognition:

```bash
python scripts/realtime_demo_seq.py
```

* Displays webcam feed with predicted mudra label.
* Recognizes both single- and two-hand mudras, including closed-hand gestures.

## Notes

* Ensure good lighting and clear hand visibility for better recognition.
* Adjust `SEQ_LENGTH` or `STRIDE` in `preprocess.py` and `realtime_demo_seq.py` for performance tuning.
* You can add more mudras by updating `mudras.json` and collecting new sequences.
