# belt-squeal-classifier

Lightweight belt squeal detector for automotive drive belts. It classifies short audio clips as **normal** or **squeal** using hand-built MFCC and spectral features with an SVM, with no librosa, numba or TensorFlow.

## How it works

1. **Load**: each `.wav` is converted to mono and resampled to 16 kHz.
2. **Features**: 37 values per clip, computed with plain NumPy/SciPy:
   - 13 MFCCs (mean and std over frames)
   - spectral centroid, bandwidth, rolloff and flatness (mean and std)
   - zero-crossing rate (mean and std) and RMS energy
3. **Classify**: `StandardScaler` followed by an RBF-kernel `SVC` (class-balanced).
4. **Split**: 70% train / 15% validation / 15% test, done per class and **grouped by source recording**, so segments or augmented copies of one recording never appear in more than one split (this prevents data leakage). `C` is tuned on the validation set; the test set is only used for the final score.

## Installation

```bash
pip install numpy scipy soundfile scikit-learn joblib
```

## Data layout

```
<RAW_DIR>/
├── normal/   *.wav
└── squeal/   *.wav
```

Set `RAW_DIR` at the top of `belt_drives.py` to your dataset folder. Files named like `clip_seg3_aug1.wav` are grouped under the base name `clip`; adjust `base_id()` if your naming differs.

## Usage

```bash
# Train (also writes model.joblib and split.json)
python belt_drives.py --train

# Evaluate on the 15% validation set (used for tuning C)
python belt_drive_validation.py

# Final score on the untouched 15% test set
python belt_drive_test.py

# Classify a single clip
python belt_drives.py --predict path/to/clip.wav
```

## Results

| Split | Files | Result |
|-------|-------|--------|
| Validation | 684 | macro-F1 0.994 |
| Test | 702 | 98.58% accuracy (692 / 702 correct) |

Dataset: 3,960 normal and 657 squeal clips (augmented).

## Limitations

- The squeal class comes from only **5 distinct source recordings** (normal: 118). The validation and test squeal sets are therefore each essentially one unseen recording, so these scores show generalisation to those recordings, not necessarily to other belts, speeds or microphone positions.
- Accuracy is flattering on imbalanced data (the test set is about 85% normal); check per-class recall in the classification report.
- More varied squeal recordings would be the most valuable improvement.

## Project files

| File | Purpose |
|------|---------|
| `belt_drives.py` | feature extraction, grouped split, training, prediction |
| `belt_drive_validation.py` | evaluates the saved model on the 15% validation split |
| `belt_drive_test.py` | final evaluation on the held-out 15% test split |
