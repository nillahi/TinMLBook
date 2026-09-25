"""
Project 1 — Keyword Spotting (training)

Usage:
  python train_kws.py --demo          # synthetic data, fast smoke test
  python train_kws.py --epochs 10     # longer demo training

Outputs:
  models/kws_float.h5
  models/kws_int8.tflite
  models/model_data.h   (for Arduino firmware)
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

NUM_CLASSES = 6  # yes, no, up, down, silence, unknown
N_MFCC = 13
N_FRAMES = 49
INPUT_SHAPE = (N_FRAMES, N_MFCC, 1)


def build_kws_model(num_classes: int = NUM_CLASSES) -> keras.Model:
    """Small SeparableConv model suitable for Cortex-M4."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=INPUT_SHAPE),
            keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.SeparableConv2D(32, (3, 3), padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def make_synthetic_dataset(n_per_class: int = 40, seed: int = 42):
    """Generate fake MFCC-like tensors so beginners can run without downloads."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label in range(NUM_CLASSES):
        # Class-dependent tone so the tiny model can learn something
        base = rng.normal(0, 0.3, size=(n_per_class, N_FRAMES, N_MFCC, 1)).astype(np.float32)
        base += 0.4 * np.sin(
            (np.arange(N_FRAMES)[None, :, None, None] + label) * 0.2
        ).astype(np.float32)
        xs.append(base)
        ys.append(np.full(n_per_class, label, dtype=np.int32))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def main():
    parser = argparse.ArgumentParser(description="Train TinyML keyword spotter")
    parser.add_argument("--demo", action="store_true", help="Fast synthetic training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "models")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    x, y = make_synthetic_dataset(n_per_class=50 if args.demo else 80)
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = build_kws_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    epochs = 1 if args.demo else args.epochs
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=16,
        verbose=1,
    )

    float_path = args.outdir / "kws_float.h5"
    model.save(float_path)
    print(f"Saved {float_path}")

    tflite_path = convert_to_int8_tflite(
        model, x_train[:80], args.outdir / "kws_int8.tflite"
    )
    tflite_to_c_array(tflite_path, args.outdir / "model_data.h", array_name="g_kws_model")
    print("Done. Copy models/model_data.h into firmware/kws_nano33/")


if __name__ == "__main__":
    main()
