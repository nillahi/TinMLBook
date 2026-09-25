# Keyword Spotting (Arduino Nano 33 BLE Sense)

Wake-word style classifier for `yes` / `no` / `up` / `down` (+ silence/unknown).

## Hardware
- Arduino Nano 33 BLE Sense (or Rev2)
- Onboard PDM microphone (no wiring)

## Train
```bash
cd training
pip install -r ../../requirements.txt
python train_kws.py --demo
# Copy models/model_data.h → firmware/kws_nano33/model_data.h
```

For a real model, collect Google Speech Commands (or your own WAVs) and replace
`make_synthetic_dataset()` with a proper MFCC pipeline (see book Ch. 10).

## Flash
1. Arduino IDE → Board: **Nano 33 BLE**
2. Library: **Arduino_TensorFlowLite**
3. Open `firmware/kws_nano33/kws_nano33.ino`, Upload
4. Serial Monitor **115200**

## Success criteria
- Serial prints `KWS ready`
- Speaking near the mic produces `Heard: <label>` lines
- After real training: >85% accuracy on held-out Speech Commands subset

## Notes
The shipped sketch uses a **feature placeholder** (energy → dummy MFCC tensor).
For production accuracy, add a real MFCC frontend or deploy an Edge Impulse
library that includes preprocessing.
