/*
 * Predictive Maintenance — vibration classifier
 *
 * Target: STM32 Nucleo / Arduino-compatible board with ADXL355 (SPI)
 *          or any 3-axis accelerometer providing float g readings.
 *
 * This sketch is structured for Arduino; on bare-metal STM32, map
 * readAccel() to your HAL SPI driver for the ADXL355.
 */

#include <SPI.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "feature_norm.h"

const char* kLabels[] = {"normal", "imbalance", "bearing"};
constexpr int kWindow = 128;
constexpr int kFeats = 6;
constexpr int kArenaSize = 20 * 1024;

alignas(16) uint8_t tensor_arena[kArenaSize];
float ax[kWindow], ay[kWindow], az[kWindow];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// --- Hardware abstraction: replace with ADXL355 SPI reads on STM32 ---
bool readAccel(float& x, float& y, float& z) {
  // Demo: synthetic idle vibration so Serial shows "normal" without sensor
  static uint32_t t = 0;
  t++;
  x = 0.02f * sinf(t * 0.05f);
  y = 0.02f * cosf(t * 0.05f);
  z = 1.0f + 0.01f * sinf(t * 0.02f);
  return true;
}

void computeFeatures(float* out6) {
  float mag[kWindow];
  float sum = 0, sum2 = 0, peak = 0;
  for (int i = 0; i < kWindow; i++) {
    mag[i] = sqrtf(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]);
    sum += mag[i];
    sum2 += mag[i] * mag[i];
    if (mag[i] > peak) peak = mag[i];
  }
  float mean = sum / kWindow;
  float rms = sqrtf(sum2 / kWindow);
  float var = 0, m4 = 0;
  for (int i = 0; i < kWindow; i++) {
    float d = mag[i] - mean;
    var += d * d;
    m4 += d * d * d * d;
  }
  float stdv = sqrtf(var / kWindow) + 1e-6f;
  out6[0] = mean;
  out6[1] = stdv;
  out6[2] = rms;
  out6[3] = peak;
  out6[4] = peak / (rms + 1e-6f);
  out6[5] = (m4 / kWindow) / (stdv * stdv * stdv * stdv);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  model = tflite::GetModel(g_pdm_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
  interpreter = &interp;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("PdM ready");
}

void loop() {
  for (int i = 0; i < kWindow; i++) {
    readAccel(ax[i], ay[i], az[i]);
    delay(1);  // ~1 kHz sampling when sensor is real
  }

  float feats[kFeats];
  computeFeatures(feats);

  // Quantize float features → int8 using training mean/std then input scale
  for (int i = 0; i < kFeats; i++) {
    float n = (feats[i] - kFeatMean[i]) / kFeatStd[i];
    // Approximate: if model expects int8, use scale from tensor
    float scale = input->params.scale;
    int zp = input->params.zero_point;
    int q = (int)roundf(n / scale) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    input->data.int8[i] = (int8_t)q;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  int best = 0;
  int8_t best_s = -128;
  for (int i = 0; i < 3; i++) {
    if (output->data.int8[i] > best_s) {
      best_s = output->data.int8[i];
      best = i;
    }
  }
  Serial.print("State: ");
  Serial.println(kLabels[best]);
  delay(200);
}
