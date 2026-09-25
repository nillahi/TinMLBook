"""
Project 4 — Environmental Anomaly Detection (BME688-style features)

Features: temperature, humidity, pressure, gas resistance
Usage: python train_env.py --demo
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

N_FEATURES = 4  # T, H, P, gas
# autoencoder bottleneck — anomaly if reconstruction error high


def synth_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    # Room-ish conditions
    t = rng.normal(22.0, 0.8, size=(n, 1))
    h = rng.normal(45.0, 5.0, size=(n, 1))
    p = rng.normal(1013.0, 2.0, size=(n, 1))
    g = rng.normal(50e3, 5e3, size=(n, 1))
    return np.hstack([t, h, p, g]).astype(np.float32)


def synth_anomaly(n: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.normal(35.0, 2.0, size=(n, 1))  # hot / VOC event
    h = rng.normal(70.0, 5.0, size=(n, 1))
    p = rng.normal(1005.0, 3.0, size=(n, 1))
    g = rng.normal(10e3, 2e3, size=(n, 1))  # low gas resistance = contamination
    return np.hstack([t, h, p, g]).astype(np.float32)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def build_autoencoder() -> keras.Model:
    inp = keras.Input(shape=(N_FEATURES,))
    z = keras.layers.Dense(8, activation="relu")(inp)
    z = keras.layers.Dense(2, activation="relu")(z)
    z = keras.layers.Dense(8, activation="relu")(z)
    out = keras.layers.Dense(N_FEATURES, activation="linear")(z)
    return keras.Model(inp, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "models")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    normal = synth_normal(400 if not args.demo else 120, rng)
    mean, std = normal.mean(0), normal.std(0) + 1e-6
    x = normalize(normal, mean, std)

    model = build_autoencoder()
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, x, epochs=1 if args.demo else args.epochs, batch_size=32, verbose=1, validation_split=0.2)

    # Threshold from training recon error
    recon = model.predict(x, verbose=0)
    err = np.mean((x - recon) ** 2, axis=1)
    threshold = float(np.percentile(err, 95))
    print(f"Anomaly MSE threshold: {threshold:.6f}")

    # Quick check
    anom = normalize(synth_anomaly(20, rng), mean, std)
    aerr = np.mean((anom - model.predict(anom, verbose=0)) ** 2, axis=1)
    print(f"Normal mean err={err.mean():.4f}  Anomaly mean err={aerr.mean():.4f}")

    model.save(args.outdir / "env_ae_float.h5")
    np.savez(args.outdir / "env_norm.npz", mean=mean, std=std, threshold=threshold)

    tflite = convert_to_int8_tflite(model, x[:80], args.outdir / "env_ae_int8.tflite")
    tflite_to_c_array(tflite, args.outdir / "model_data.h", "g_env_model")

    hdr = args.outdir / "env_norm.h"
    hdr.write_text(
        "#pragma once\n"
        "const float kEnvMean[4] = {%s};\n"
        "const float kEnvStd[4] = {%s};\n"
        "const float kAnomalyThreshold = %.8ff;\n"
        % (
            ", ".join(f"{v:.6f}f" for v in mean),
            ", ".join(f"{v:.6f}f" for v in std),
            threshold,
        )
    )
    print(f"Wrote {hdr}")


if __name__ == "__main__":
    main()
