"""
Scores the saved model on the held-out split created by belt_drives.py --train.

    python belt_drive_validation.py            # 15% TEST set (default)
    python belt_drive_validation.py --set val  # 15% validation set

The test set was never used for fitting or for choosing C, so this is the
number to report. Run it once you're done tuning.
"""

import os
import json
import argparse
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from belt_drives import extract_features, LABELS, MODEL_FILE, SPLIT_FILE


def label_from_path(path):
    """Class label comes from the parent folder name (normal/ or squeal/)."""
    return LABELS[os.path.basename(os.path.dirname(path))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["test", "val"], default="test")
    args = parser.parse_args()

    for required in (MODEL_FILE, SPLIT_FILE):
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"'{required}' not found -- run `python belt_drives.py --train` first."
            )

    with open(SPLIT_FILE) as f:
        manifest = json.load(f)
    paths = manifest[args.set]

    print("==========================================")
    print(f"   BELT SQUEAL EVALUATION ({args.set.upper()} SET)")
    print("==========================================")
    print(f"Files in split: {len(paths)}")

    model = joblib.load(MODEL_FILE)

    X, y_true, files = [], [], []
    for p in paths:
        try:
            X.append(extract_features(p))
            y_true.append(label_from_path(p))
            files.append(p)
        except Exception as e:
            print(f"[SKIP] {p}\n       Reason: {e}")

    if not X:
        raise RuntimeError("No files could be processed.")

    X, y_true = np.array(X), np.array(y_true)
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=["normal", "squeal"], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix:")
    print("              Predicted")
    print("              Normal  Squeal")
    print(f"Actual Normal   {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"Actual Squeal   {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    print("\n========== MISCLASSIFIED FILES ==========")
    wrong = [(p, a, b) for p, a, b in zip(files, y_true, y_pred) if a != b]
    names = {v: k for k, v in LABELS.items()}
    for p, a, b in wrong:
        print(f"[WRONG] Actual={names[a]:<6} Predicted={names[b]:<6} {os.path.basename(p)}")
    if not wrong:
        print("None")

    print("\n==========================================")
    print(f"Files: {len(y_true)}   Correct: {len(y_true) - len(wrong)}   "
          f"Wrong: {len(wrong)}   Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()