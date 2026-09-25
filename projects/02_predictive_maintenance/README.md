# Predictive Maintenance (vibration)

Classify machine vibration into `normal` / `imbalance` / `bearing`.

## Hardware
- STM32 Nucleo (or Arduino with enough RAM) + **ADXL355** (SPI), **or**
- Any 3-axis accelerometer — replace `readAccel()` in the sketch
- Book target: industrial-style vibration sensing

## Train
```bash
cd training
python train_pdm.py --demo
# copy models/model_data.h and models/feature_norm.h → firmware/pdm_stm32/
```

## Flash
1. Install board support + Arduino_TensorFlowLite
2. Open `firmware/pdm_stm32/pdm_stm32.ino`
3. Wire ADXL355 (CS/SCLK/MOSI/MISO) and implement SPI in `readAccel`
4. Serial **115200** → `State: normal|imbalance|bearing`

## Success criteria
- Demo mode without sensor prints a state every ~200 ms
- With real ADXL355 data + retraining: clear separation of fault classes

## Note
Stub `readAccel()` synthesizes idle vibration so you can verify the TFLM
path before connecting hardware.
