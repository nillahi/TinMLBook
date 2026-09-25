/*
 * Gesture Recognition — Arduino Nano 33 BLE Sense
 *
 * IMPORTANT: Nano 33 BLE Sense (original) uses LSM9DS1, not LSM6DSOX.
 * Rev2 boards use BMI270 + BMM150 — use Arduino_BMI270_BMM150 instead.
 *
 * Library (original Sense): Arduino_LSM9DS1
 */

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

const char* kLabels[] = {"idle", "circle", "swipe_x", "swipe_y", "tap"};
constexpr int kSeqLen = 64;
constexpr int kAxes = 6;
constexpr int kArenaSize = 40 * 1024;

alignas(16) uint8_t tensor_arena[kArenaSize];
float window[kSeqLen][kAxes];
int win_idx = 0;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!IMU.begin()) {
    Serial.println("Failed to init LSM9DS1 — check board revision");
    while (1) delay(1000);
  }
  Serial.print("Acc sample rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  model = tflite::GetModel(g_gesture_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
  interpreter = &interp;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Gesture ready. Move the board.");
}

void loop() {
  float ax, ay, az, gx, gy, gz;
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    return;
  }
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  window[win_idx][0] = ax;
  window[win_idx][1] = ay;
  window[win_idx][2] = az;
  window[win_idx][3] = gx;
  window[win_idx][4] = gy;
  window[win_idx][5] = gz;
  win_idx++;

  if (win_idx < kSeqLen) {
    return;
  }
  win_idx = 0;

  // Quantize window into int8 input (approximate using tensor scale)
  float scale = input->params.scale;
  int zp = input->params.zero_point;
  int8_t* dst = input->data.int8;
  int n = 0;
  for (int t = 0; t < kSeqLen; t++) {
    for (int a = 0; a < kAxes; a++) {
      int q = (int)roundf(window[t][a] / scale) + zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      dst[n++] = (int8_t)q;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  int best = 0;
  int8_t best_s = -128;
  for (int i = 0; i < 5; i++) {
    if (output->data.int8[i] > best_s) {
      best_s = output->data.int8[i];
      best = i;
    }
  }
  Serial.print("Gesture: ");
  Serial.println(kLabels[best]);
}
