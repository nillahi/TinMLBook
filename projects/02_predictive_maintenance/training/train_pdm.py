"""
Project 2 — Predictive Maintenance (vibration anomaly / fault classification)

Usage:
  python train_pdm.py --demo
Outputs models under ./models/
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

WINDOW = 128  # acceleration samples (e.g. 1 kHz → 128 ms)
N_FEATURES = 6  # mean, std, rms, peak, crest, kurtosis-ish per axis collapsed
NUM_CLASSES = 3  # normal, imbalance, bearing


def extract_features(window: np.ndarray) -> np.ndarray:
    """window: (WINDOW, 3) float32 accel → (N_FEATURES,)"""
    x, y, z = window[:, 0], window[:, 1], window[:, 2]
    mag = np.sqrt(x * x + y * y + z * z)
    feats = np.array(
        [
            mag.mean(),
            mag.std() + 1e-6,
            np.sqrt((mag**2).mean()),
            mag.max(),
            mag.max() / (np.sqrt((mag**2).mean()) + 1e-6),
            ((mag - mag.mean()) ** 4).mean() / ((mag.std() + 1e-6) ** 4),
        ],
        dtype=np.float32,
    )
    return feats


def synth_windows(n: int, label: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, WINDOW, dtype=np.float32)
    base = 0.05 * rng.normal(size=(n, WINDOW, 3)).astype(np.float32)
    if label == 0:  # normal
        base += 0.02 * np.sin(2 * np.pi * 50 * t)[None, :, None]
    elif label == 1:  # imbalance — strong 1x harmonic
        base += 0.25 * np.sin(2 * np.pi * 30 * t)[None, :, None]
    else:  # bearing — high-freq bursts
        base += 0.15 * rng.normal(size=(n, WINDOW, 3)).astype(np.float32)
        base[:, ::8, :] += 0.4
    return base


def build_dataset(n_per: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for label in range(NUM_CLASSES):
        windows = synth_windows(n_per, label, rng)
        for w in windows:
            xs.append(extract_features(w))
            ys.append(label)
    x = np.stack(xs)
    y = np.array(ys, dtype=np.int32)
    # normalize
    mean, std = x.mean(0), x.std(0) + 1e-6
    x = (x - mean) / std
    idx = rng.permutation(len(x))
    return x[idx], y[idx], mean, std


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(N_FEATURES,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "models")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    x, y, mean, std = build_dataset(80 if not args.demo else 40)
    np.savez(args.outdir / "feature_norm.npz", mean=mean, std=std)

    split = int(0.8 * len(x))
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x[:split], y[:split], validation_data=(x[split:], y[split:]),
              epochs=1 if args.demo else args.epochs, batch_size=16, verbose=1)
    model.save(args.outdir / "pdm_float.h5")

    tflite = convert_to_int8_tflite(model, x[:80], args.outdir / "pdm_int8.tflite")
    tflite_to_c_array(tflite, args.outdir / "model_data.h", "g_pdm_model")

    # Emit C norms for firmware
    hdr = args.outdir / "feature_norm.h"
    hdr.write_text(
        "#pragma once\n"
        "const float kFeatMean[%d] = {%s};\n"
        "const float kFeatStd[%d] = {%s};\n"
        % (
            N_FEATURES,
            ", ".join(f"{v:.6f}f" for v in mean),
            N_FEATURES,
            ", ".join(f"{v:.6f}f" for v in std),
        )
    )
    print(f"Wrote {hdr}")


if __name__ == "__main__":
    main()
