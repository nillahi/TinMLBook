"""
Project 3 — Gesture Recognition (IMU)

Classes: circle, swipe_x, swipe_y, tap, idle
Usage: python train_gesture.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[3]  # TinMLBook/
sys.path.insert(0, str(ROOT))
from common.quantize_utils import convert_to_int8_tflite, tflite_to_c_array  # noqa: E402

SEQ_LEN = 64  # timesteps
N_AXES = 6    # ax,ay,az,gx,gy,gz
NUM_CLASSES = 5
LABELS = ["idle", "circle", "swipe_x", "swipe_y", "tap"]


def synth_seq(label: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, SEQ_LEN, dtype=np.float32)
    seq = 0.05 * rng.normal(size=(SEQ_LEN, N_AXES)).astype(np.float32)
    if label == 1:  # circle in xy accel
        seq[:, 0] += np.cos(t)
        seq[:, 1] += np.sin(t)
    elif label == 2:
        seq[:, 0] += np.sin(t * 2)
    elif label == 3:
        seq[:, 1] += np.sin(t * 2)
    elif label == 4:  # tap — spike mid-window
        seq[SEQ_LEN // 2, 2] += 2.5
    return seq


def build_dataset(n_per=40, seed=1):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for lab in range(NUM_CLASSES):
        for _ in range(n_per):
            xs.append(synth_seq(lab, rng))
            ys.append(lab)
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int32)
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_AXES)),
            keras.layers.Conv1D(16, 5, activation="relu", padding="same"),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "models")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    x, y = build_dataset(50 if not args.demo else 30)
    split = int(0.8 * len(x))
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x[:split], y[:split], validation_data=(x[split:], y[split:]),
              epochs=1 if args.demo else args.epochs, batch_size=16, verbose=1)
    model.save(args.outdir / "gesture_float.h5")
    tflite = convert_to_int8_tflite(model, x[:60], args.outdir / "gesture_int8.tflite")
    tflite_to_c_array(tflite, args.outdir / "model_data.h", "g_gesture_model")
    print("Labels:", LABELS)


if __name__ == "__main__":
    main()
