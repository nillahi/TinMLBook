/*
 * Environmental Anomaly Detection — BME688 (or BME680)
 *
 * Wiring (I2C): VIN→3V3, GND→GND, SCL→SCL, SDA→SDA
 * Library: BME68x Sensor library (Bosch) or Adafruit BME680
 *
 * Uses an autoencoder: high reconstruction error ⇒ anomaly.
 */

#include <Wire.h>
#include <Adafruit_BME680.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "env_norm.h"

constexpr int kArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];

Adafruit_BME680 bme;  // I2C
bool have_sensor = false;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  have_sensor = bme.begin(0x76) || bme.begin(0x77);
  if (!have_sensor) {
    Serial.println("BME680/688 not found — using demo values");
  } else {
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150);
    Serial.println("BME68x OK");
  }

  model = tflite::GetModel(g_env_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
  interpreter = &interp;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Env anomaly detector ready");
}

void readEnv(float& t, float& h, float& p, float& g) {
  if (have_sensor && bme.performReading()) {
    t = bme.temperature;
    h = bme.humidity;
    p = bme.pressure / 100.0f;
    g = bme.gas_resistance;
  } else {
    // Demo "normal" room
    t = 22.0f;
    h = 45.0f;
    p = 1013.0f;
    g = 50000.0f;
  }
}

void loop() {
  float t, h, p, g;
  readEnv(t, h, p, g);

  float feats[4] = {
      (t - kEnvMean[0]) / kEnvStd[0],
      (h - kEnvMean[1]) / kEnvStd[1],
      (p - kEnvMean[2]) / kEnvStd[2],
      (g - kEnvMean[3]) / kEnvStd[3],
  };

  float scale = input->params.scale;
  int zp = input->params.zero_point;
  for (int i = 0; i < 4; i++) {
    int q = (int)roundf(feats[i] / scale) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    input->data.int8[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    delay(1000);
    return;
  }

  // Dequantize output and compute MSE in normalized space
  float oscale = output->params.scale;
  int ozp = output->params.zero_point;
  float mse = 0;
  for (int i = 0; i < 4; i++) {
    float recon = (output->data.int8[i] - ozp) * oscale;
    float d = feats[i] - recon;
    mse += d * d;
  }
  mse /= 4.0f;

  Serial.print("T=");
  Serial.print(t);
  Serial.print(" H=");
  Serial.print(h);
  Serial.print(" MSE=");
  Serial.print(mse, 5);
  if (mse > kAnomalyThreshold) {
    Serial.println("  ANOMALY");
  } else {
    Serial.println("  ok");
  }
  delay(2000);
}
