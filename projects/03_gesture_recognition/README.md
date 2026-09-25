# Gesture Recognition (IMU)

Recognize hand/board gestures: idle, circle, swipe_x, swipe_y, tap.

## Hardware
- **Arduino Nano 33 BLE Sense** → IMU = **LSM9DS1** (`Arduino_LSM9DS1`)
- **Nano 33 BLE Sense Rev2** → BMI270 (`Arduino_BMI270_BMM150`) — adapt includes
- Do **not** use `Arduino_LSM6DSOX` on the classic Sense board (common book mistake)

## Train
```bash
cd training
python train_gesture.py --demo
cp models/model_data.h ../firmware/gesture_nano33/
```

## Flash
1. Board: Nano 33 BLE
2. Libraries: Arduino_LSM9DS1 + Arduino_TensorFlowLite
3. Upload `firmware/gesture_nano33/gesture_nano33.ino`
4. Serial **115200** → `Gesture: <label>`

## Success criteria
- `Gesture ready` on boot
- Distinct motions produce changing labels after you retrain on your own traces

## Collect your own data
Log CSV of `ax,ay,az,gx,gy,gz` over Serial while performing each gesture,
then replace `synth_seq()` in training with real windows.
