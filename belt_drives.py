"""
Belt Squeal Classifier -- no librosa, no numba, no TensorFlow.

Uses only: numpy, scipy, soundfile, scikit-learn, joblib.

Directory structure expected:
    <RAW_DIR>/
    ├── normal/   *.wav
    └── squeal/   *.wav

Data split (done per class, grouped by source recording):
    70% train      -> model is fitted on this
    15% validation -> used to pick the SVM's C value
    15% test       -> never touched in training; scored by belt_drive_validation.py

Install:
    pip install numpy scipy soundfile scikit-learn joblib

Usage:
    python belt_drives.py --train
    python belt_drives.py --predict path\\to\\clip.wav
"""

import os
import re
import sys
import glob
import json
import argparse
from collections import Counter
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.fftpack import dct
import joblib

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==========================================
# CONFIG
# ==========================================
RAW_DIR = r"C:\Users\nangu\PycharmProjects\belt squeal\data\augmented"   # must contain normal/ and squeal/ subfolders
LABELS = {"normal": 0, "squeal": 1}

TARGET_SR = 16000
FRAME_LEN = 0.025      # 25ms frames
FRAME_HOP = 0.010      # 10ms hop
N_MEL_FILTERS = 26
N_MFCC = 13

MODEL_FILE = "model.joblib"
SPLIT_FILE = "split.json"          # remembers which file went to train/val/test

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train, validation, test
SPLIT_SEED = 42
C_GRID = [0.1, 1.0, 10.0, 100.0]       # candidates tuned on the validation set


# ==========================================
# AUDIO LOADING (no librosa)
# ==========================================
def load_wav_mono(path, target_sr=TARGET_SR):
    """Read a wav file, downmix to mono, resample to target_sr."""
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)

    if sr != target_sr:
        g = gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g)

    return y, target_sr


# ==========================================
# HAND-BUILT MFCC + SPECTRAL FEATURES
# ==========================================
def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700.0)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595.0) - 1)


def mel_filterbank(sr, n_fft, n_filters=N_MEL_FILTERS):
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fbank


def compute_mfcc(y, sr, n_mfcc=N_MFCC):
    frame_len = int(FRAME_LEN * sr)
    hop_len = int(FRAME_HOP * sr)
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2

    # pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # framing
    n_frames = 1 + (len(y) - frame_len) // hop_len
    if n_frames < 1:
        n_frames = 1
        y = np.pad(y, (0, frame_len - len(y)))

    frames = np.stack([
        y[i * hop_len: i * hop_len + frame_len]
        for i in range(n_frames)
        if i * hop_len + frame_len <= len(y)
    ]) if n_frames > 0 else np.zeros((1, frame_len))

    if frames.shape[0] == 0:
        frames = np.zeros((1, frame_len))

    window = np.hamming(frame_len)
    frames = frames * window

    # power spectrum
    mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))
    power = (1.0 / n_fft) * (mag ** 2)

    # mel filterbank energies
    fbank = mel_filterbank(sr, n_fft)
    mel_energy = power @ fbank.T
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel = np.log(mel_energy)

    # DCT -> MFCC
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    return mfcc, mag, sr, n_fft


def spectral_stats(mag, sr, n_fft):
    """Extra spectral descriptors, per-frame, later averaged."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    power = mag ** 2 + 1e-12
    power_sum = power.sum(axis=1, keepdims=True)

    centroid = (freqs[None, :] * power).sum(axis=1) / power_sum.squeeze()
    bandwidth = np.sqrt(
        ((freqs[None, :] - centroid[:, None]) ** 2 * power).sum(axis=1) / power_sum.squeeze()
    )

    cumulative = np.cumsum(power, axis=1)
    rolloff_thresh = 0.85 * power_sum.squeeze()
    rolloff = np.array([
        freqs[np.searchsorted(cumulative[i], rolloff_thresh[i])]
        if np.searchsorted(cumulative[i], rolloff_thresh[i]) < len(freqs) else freqs[-1]
        for i in range(mag.shape[0])
    ])

    flatness = np.exp(np.mean(np.log(power), axis=1)) / (np.mean(power, axis=1))

    return centroid, bandwidth, rolloff, flatness


def zero_crossing_rate(y, frame_len, hop_len):
    n_frames = 1 + max(0, (len(y) - frame_len) // hop_len)
    zcr = []
    for i in range(max(n_frames, 1)):
        start = i * hop_len
        frame = y[start:start + frame_len]
        if len(frame) < 2:
            zcr.append(0.0)
            continue
        zcr.append(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
    return np.array(zcr)


def extract_features(path):
    """One fixed-length feature vector per audio file (mean+std pooling
    over frames), covering MFCCs + basic spectral descriptors + ZCR."""
    y, sr = load_wav_mono(path)

    if len(y) == 0:
        raise ValueError("empty audio")

    mfcc, mag, sr, n_fft = compute_mfcc(y, sr)
    centroid, bandwidth, rolloff, flatness = spectral_stats(mag, sr, n_fft)

    frame_len = int(FRAME_LEN * sr)
    hop_len = int(FRAME_HOP * sr)
    zcr = zero_crossing_rate(y, frame_len, hop_len)

    rms = np.sqrt(np.mean(y ** 2))

    feature_parts = [
        mfcc.mean(axis=0), mfcc.std(axis=0),
        [centroid.mean(), centroid.std()],
        [bandwidth.mean(), bandwidth.std()],
        [rolloff.mean(), rolloff.std()],
        [flatness.mean(), flatness.std()],
        [zcr.mean(), zcr.std()],
        [rms],
    ]
    return np.concatenate([np.atleast_1d(p) for p in feature_parts])


# ==========================================
# DATASET BUILDING
# ==========================================
def validate_raw_dir():
    missing = []
    for label in LABELS:
        folder = os.path.join(RAW_DIR, label)
        if not os.path.isdir(folder):
            missing.append(folder)
    if missing:
        raise FileNotFoundError(
            "Expected folders not found:\n  " + "\n  ".join(missing)
        )


def build_dataset():
    validate_raw_dir()
    X, y, files = [], [], []

    for label_name, label_id in LABELS.items():
        paths = sorted(glob.glob(os.path.join(RAW_DIR, label_name, "*.wav")))
        print(f"Processing '{label_name}': {len(paths)} files")
        for path in paths:
            try:
                feat = extract_features(path)
                X.append(feat)
                y.append(label_id)
                files.append(path)
            except Exception as e:
                print(f"  [skip] {path}: {e}")

    if not X:
        raise RuntimeError("No usable files found -- check RAW_DIR and subfolders.")

    return np.array(X), np.array(y), files


# ==========================================
# 70 / 15 / 15 SPLIT (grouped, to avoid leakage)
# ==========================================
def base_id(path):
    """Strip segment/augmentation suffixes to find the original source name.
    e.g. 'belt1_seg3_aug2.wav' and 'belt1_seg4.wav' both -> 'belt1'."""
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"(_seg\d+|_aug\d+|_orig)+$", "", name)


def _grouped_assign(y, groups, fractions, seed, tries=300):
    """Per-class grouped split. For each class, shuffle its recordings many
    times and, for each shuffle, try every cut point (first a groups -> train,
    next b -> val, rest -> test, each non-empty). Keep the cut whose file-count
    proportions are closest to the target. Returns None if any class has fewer
    than 3 distinct recordings (one per split is impossible)."""
    rng = np.random.RandomState(seed)
    assign = np.zeros(len(y), dtype=int)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        counts = Counter(groups[idx])
        gids = sorted(counts)
        n = len(gids)
        if n < 3:
            return None
        total = len(idx)

        best_err, best_map = None, None
        for _ in range(tries):
            order = list(gids)
            rng.shuffle(order)
            sizes = np.array([counts[g] for g in order])
            csum = np.concatenate([[0], np.cumsum(sizes)])
            for a in range(1, n - 1):
                for b in range(a + 1, n):
                    f_tr = csum[a] / total
                    f_va = (csum[b] - csum[a]) / total
                    f_te = (total - csum[b]) / total
                    err = (abs(f_tr - fractions[0]) + abs(f_va - fractions[1])
                           + abs(f_te - fractions[2]))
                    if best_err is None or err < best_err:
                        best_err = err
                        best_map = {g: (0 if k < a else (1 if k < b else 2))
                                    for k, g in enumerate(order)}
        for i in idx:
            assign[i] = best_map[groups[i]]
    return assign


def _is_valid(y, assign):
    """Every split must contain both classes."""
    return all(len(np.unique(y[assign == s])) == 2 for s in range(3))


def _file_level_assign(y, fractions, seed):
    """Fallback: plain stratified split by file (can leak across recordings)."""
    rng = np.random.RandomState(seed)
    assign = np.zeros(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        a = int(round(fractions[0] * len(idx)))
        b = int(round((fractions[0] + fractions[1]) * len(idx)))
        assign[idx[a:b]] = 1
        assign[idx[b:]] = 2
    return assign


def split_indices(y, files, fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED):
    """Return an array of 0 (train) / 1 (val) / 2 (test) per file.

    Splits per class and by source group (so segments/augmentations of one
    recording stay together). Needs at least 3 distinct recordings per class.
    Falls back to a file-level split (with a warning) if there are too few
    distinct recordings for a grouped split to work."""
    groups = np.array([base_id(f) for f in files])

    print("\n--- SOURCE RECORDINGS (distinct base names) ---")
    for cls_name, cls_id in LABELS.items():
        m = y == cls_id
        print(f"  {cls_name}: {m.sum()} files from {len(set(groups[m]))} recording(s)")

    assign = _grouped_assign(y, groups, fractions, seed)
    if assign is not None and _is_valid(y, assign):
        return assign

    print("\n[WARNING] Too few distinct recordings in at least one class to build a "
          "grouped 70/15/15 split.")
    print("          Falling back to a per-FILE stratified split. Segments/augmented "
          "copies of the same recording can now sit in both train and test, so "
          "test accuracy will be OPTIMISTIC. Add more original recordings, or check "
          "that base_id() matches your file naming.")
    fallback = _file_level_assign(y, fractions, seed)
    if not _is_valid(y, fallback):
        raise ValueError("Not enough files per class to make a 70/15/15 split "
                         "(each split needs at least one normal and one squeal file).")
    return fallback


# ==========================================
# TRAIN
# ==========================================
def build_pipeline(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, probability=True, class_weight="balanced")),
    ])


def run_training():
    print("\n--- EXTRACTING FEATURES ---")
    X, y, files = build_dataset()
    print(f"Dataset: {len(y)} files ({(y == 1).sum()} squeal, {(y == 0).sum()} normal), "
          f"{X.shape[1]} features each")

    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes present to train.")

    # ---- split ----
    assign = split_indices(y, files)
    names = ["train", "val", "test"]
    print("\n--- SPLIT (70/15/15, grouped by source recording) ---")
    for s, name in enumerate(names):
        m = assign == s
        print(f"{name:>5}: {m.sum():5d} files "
              f"({m.sum() / len(y) * 100:4.1f}%)  "
              f"normal={(y[m] == 0).sum()}  squeal={(y[m] == 1).sum()}")
        if len(np.unique(y[m])) < 2:
            raise ValueError(
                f"'{name}' split is missing a class. You need more distinct "
                f"source recordings per class for a 70/15/15 split."
            )

    tr, va = assign == 0, assign == 1
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    # ---- pick C on the validation set ----
    print("\n--- TUNING C ON VALIDATION SET ---")
    best_C, best_f1, best_model = None, -1.0, None
    for C in C_GRID:
        model = build_pipeline(C).fit(X_tr, y_tr)
        f1 = f1_score(y_va, model.predict(X_va), average="macro")
        print(f"  C={C:<6} validation macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_C, best_f1, best_model = C, f1, model

    print(f"\nBest C = {best_C}")
    y_va_pred = best_model.predict(X_va)
    print("\nValidation report:")
    print(classification_report(y_va, y_va_pred, target_names=["normal", "squeal"], zero_division=0))
    print("Validation confusion matrix:")
    print(confusion_matrix(y_va, y_va_pred, labels=[0, 1]))

    # ---- save model (trained on the 70% train split only) + split manifest ----
    joblib.dump(best_model, MODEL_FILE)
    manifest = {name: [files[i] for i in np.where(assign == s)[0]]
                for s, name in enumerate(names)}
    with open(SPLIT_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nModel (trained on train split only) saved to '{MODEL_FILE}'")
    print(f"Split lists saved to '{SPLIT_FILE}'")
    print("Run belt_drive_validation.py to score the untouched 15% test set.")


# ==========================================
# PREDICT
# ==========================================
def predict_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"'{MODEL_FILE}' not found -- run --train first.")

    model = joblib.load(MODEL_FILE)
    feat = extract_features(path).reshape(1, -1)
    prob_squeal = model.predict_proba(feat)[0, 1]
    verdict = "SQUEAL" if prob_squeal >= 0.5 else "NORMAL"

    print(f"\nFile: {path}")
    print(f"P(squeal) = {prob_squeal:.3f}")
    print(f"Verdict: {verdict}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str)
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        print("No arguments given -- running --train by default.")
        args.train = True

    try:
        if args.train:
            run_training()
        if args.predict:
            predict_file(args.predict)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)