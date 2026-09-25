/*
 * Keyword Spotting — Arduino Nano 33 BLE Sense
 *
 * Board: Arduino Nano 33 BLE Sense (nRF52840) or BLE Sense Rev2
 * Library: Arduino_TensorFlowLite (or EloquentTinyML / tflite-micro)
 *
 * Setup:
 *   1. Tools → Board → Arduino Mbed OS Nano Boards → Nano 33 BLE
 *   2. Install "Arduino_TensorFlowLite" from Library Manager
 *   3. Place model_data.h (from training/) next to this sketch
 *   4. Upload; open Serial Monitor at 115200
 *
 * Expected serial: "Heard: yes" / "no" / "up" / "down" when you speak
 * near the onboard mic (demo model is synthetic — retrain on Speech Commands).
 */

#include <PDM.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// Labels must match training order
const char* kLabels[] = {"yes", "no", "up", "down", "silence", "unknown"};
constexpr int kNumLabels = 6;

// 16 kHz, 1 second ≈ 16000 samples; we use a shorter window for responsiveness
constexpr int kSampleRate = 16000;
constexpr int kAudioLength = 16000;
constexpr int kArenaSize = 60 * 1024;

alignas(16) uint8_t tensor_arena[kArenaSize];
short sampleBuffer[256];
volatile int samplesRead = 0;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  model = tflite::GetModel(g_kws_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch");
    while (1) delay(1000);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed — increase kArenaSize");
    while (1) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, kSampleRate)) {
    Serial.println("Failed to start PDM microphone");
    while (1) delay(1000);
  }
  PDM.setGain(20);

  Serial.println("KWS ready. Speak near the mic.");
  Serial.print("Input shape dims: ");
  Serial.println(input->dims->size);
}

// Minimal MFCC placeholder: real deployments should use a proper MFCC pipeline
// (see book Ch. 10 / Edge Impulse processing block). Here we fill a fixed
// spectrogram-sized buffer from RMS energy so the sketch compiles and runs.
void fillDummyFeaturesFromAudio(int16_t energy) {
  // Expect int8 input; scale a constant pattern by energy for demo
  int n = input->bytes;
  int8_t* data = input->data.int8;
  for (int i = 0; i < n; i++) {
    data[i] = (int8_t)((energy / 64) + (i % 7) - 3);
  }
}

void loop() {
  static int16_t pcm[kAudioLength];
  static int filled = 0;

  if (samplesRead > 0) {
    for (int i = 0; i < samplesRead && filled < kAudioLength; i++) {
      pcm[filled++] = sampleBuffer[i];
    }
    samplesRead = 0;
  }

  if (filled < kAudioLength) {
    return;
  }
  filled = 0;

  // Simple energy gate — skip silence
  long sum = 0;
  for (int i = 0; i < kAudioLength; i++) {
    sum += abs(pcm[i]);
  }
  int energy = (int)(sum / kAudioLength);
  if (energy < 50) {
    return;
  }

  fillDummyFeaturesFromAudio(energy);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  int best = 0;
  int8_t best_score = -128;
  for (int i = 0; i < kNumLabels; i++) {
    int8_t s = output->data.int8[i];
    if (s > best_score) {
      best_score = s;
      best = i;
    }
  }

  Serial.print("Heard: ");
  Serial.print(kLabels[best]);
  Serial.print("  score=");
  Serial.println(best_score);
  delay(300);
}
