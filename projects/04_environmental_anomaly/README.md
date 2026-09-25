# Environmental Anomaly Detection (BME688)

Autoencoder on temperature / humidity / pressure / gas resistance.
High reconstruction MSE → `ANOMALY`.

## Hardware
- MCU with I2C (Nano 33 BLE, ESP32, STM32, …)
- **Bosch BME688** or **BME680**
- Wiring: VIN→3V3, GND, SCL, SDA (addr `0x76` or `0x77`)

## Train
```bash
cd training
python train_env.py --demo
cp models/model_data.h models/env_norm.h ../firmware/env_bme688/
```

## Flash
1. Libraries: **Adafruit BME680** + Arduino_TensorFlowLite
2. Upload `firmware/env_bme688/env_bme688.ino`
3. Serial **115200** → `ok` or `ANOMALY`

## Success criteria
- Without sensor: demo room values print `ok`
- After training on your site’s “normal” air: VOC/heat events trip `ANOMALY`

## Tip
Retrain the autoencoder on several days of *your* normal environment;
the book’s synthetic threshold will not match a new room.
