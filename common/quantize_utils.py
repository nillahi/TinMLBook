"""Shared helpers for TinyML companion projects."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf


def convert_to_int8_tflite(
    model: tf.keras.Model,
    representative_data: np.ndarray,
    output_path: Path,
    input_type: tf.DType = tf.int8,
    output_type: tf.DType = tf.int8,
) -> Path:
    """Convert a Keras model to full-integer int8 TFLite.

    representative_data: float32 array shaped like model inputs, used to
    calibrate quantization ranges. Use a small batch (50-200 samples).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def representative_dataset():
        for i in range(min(len(representative_data), 100)):
            yield [representative_data[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = input_type
    converter.inference_output_type = output_type

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Wrote {output_path} ({len(tflite_model)} bytes)")
    return output_path


def tflite_to_c_array(
    tflite_path: Path,
    header_path: Path,
    array_name: str = "g_model",
) -> Path:
    """Embed a .tflite file as a C header byte array for Arduino/TFLM."""
    tflite_path = Path(tflite_path)
    header_path = Path(header_path)
    data = tflite_path.read_bytes()
    header_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "// Auto-generated from %s — do not edit by hand" % tflite_path.name,
        "#pragma once",
        "#include <cstdint>",
        "",
        "alignas(16) const unsigned char %s[] = {" % array_name,
    ]
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        hexes = ", ".join("0x%02x" % b for b in chunk)
        lines.append("  %s," % hexes)
    lines.append("};")
    lines.append("const unsigned int %s_len = %d;" % (array_name, len(data)))
    lines.append("")
    header_path.write_text("\n".join(lines))
    print(f"Wrote {header_path} ({len(data)} bytes embedded)")
    return header_path


def run_tflite_int8(model_path: Path, input_int8: np.ndarray) -> np.ndarray:
    """Run an int8 TFLite model; returns raw int8 output vector."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], input_int8.astype(np.int8))
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])
