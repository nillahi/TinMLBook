# TinMLBook — Companion Code

Runnable projects for **TinyML on Microcontrollers** by Nader Ilahi.

**Repository:** https://github.com/nillahi/TinMLBook

This repo contains **code only** (training scripts + firmware sketches).  
Book LaTeX/PDF sources are not published here.

## Projects

| # | Project | Board / sensor | Folder |
|---|---------|----------------|--------|
| 1 | Keyword Spotting | Nano 33 BLE Sense (PDM mic) | `projects/01_keyword_spotting` |
| 2 | Predictive Maintenance | STM32-class + ADXL355 (or any IMU) | `projects/02_predictive_maintenance` |
| 3 | Gesture Recognition | Nano 33 BLE Sense (**LSM9DS1**) | `projects/03_gesture_recognition` |
| 4 | Environmental Anomaly | BME688 / BME680 (I2C) | `projects/04_environmental_anomaly` |

## Quick start (training, no hardware)

```bash
git clone git@github.com:nillahi/TinMLBook.git
cd TinMLBook
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python projects/01_keyword_spotting/training/train_kws.py --demo
python projects/02_predictive_maintenance/training/train_pdm.py --demo
python projects/03_gesture_recognition/training/train_gesture.py --demo
python projects/04_environmental_anomaly/training/train_env.py --demo
```

Each script writes `models/*.tflite` and a `model_data.h` you copy into the matching firmware folder.

## Firmware

1. Install [Arduino IDE](https://www.arduino.cc/en/software) (or Arduino CLI).
2. Install **Arduino_TensorFlowLite** (Library Manager).
3. Install board + sensor libraries listed in each project `README.md`.
4. Replace stub `model_data.h` with the file from training.
5. Upload and open Serial Monitor at **115200**.

## Shared helpers

`common/quantize_utils.py` — int8 TFLite conversion and C array embedding.

## Hardware note

Sketches are written against the APIs named above. We dry-run Python training in CI-less local checks; **physical board flash is not guaranteed** until you verify on your hardware. Stub sensors (PdM, Env) let you exercise the TFLM path without a breakout.

## License

MIT — see `LICENSE`.
