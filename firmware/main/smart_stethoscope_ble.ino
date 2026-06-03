/*
  Smart Stethoscope ESP32-S3 BLE Firmware
  Sends sensor data from ESP32-S3 to Raspberry Pi by BLE.

  Updated from original:
  - SPH0645LM4H-B mics now read via I2S (replaces fake heart_sound / ambient_sound)
  - MAX30102 now read via I2C (replaces simulated heart_rate and spo2)
  - MPU-6050 now read via I2C (replaces simulated imu_x, imu_y, imu_z)
  - ECG R-peak detector added for real heart_rate from AD8232
  - All original pins, JSON fields, and BLE structure kept unchanged

  Required libraries (Arduino Library Manager):
    SparkFun MAX3010x Pulse and Proximity Sensor Library
    MPU6050 by Electronic Cats

  BLE Device Name: SmartStethoscope_ESP32
*/

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Wire.h>
#include <driver/i2s.h>
#include <MAX30105.h>
#include <spo2_algorithm.h>
#include <MPU6050.h>

#define DEVICE_NAME "SmartStethoscope_ESP32"

#define SERVICE_UUID        "7b4d0001-8a7b-4d2b-9b41-000000000001"
#define CHARACTERISTIC_UUID "7b4d0002-8a7b-4d2b-9b41-000000000001"

// Original ADC pins — unchanged
#define ECG_PIN       4
#define PRESSURE_PIN  5
#define PIEZO1_PIN    6
#define PIEZO2_PIN    7

// I2S pins for SPH0645LM4H-B microphones
// Chest mic  : SELECT pin = GND  -> left  I2S channel
// Ambient mic: SELECT pin = VDD  -> right I2S channel
// Both mics share the same SCK, WS, and DATA lines
#define I2S_SCK  8    // BCLK
#define I2S_WS   9    // LRCLK
#define I2S_DIN  10   // data in

// I2C pins — MAX30102 and MPU-6050 share the same bus
#define I2C_SDA  11
#define I2C_SCL  12

// I2S driver settings
#define I2S_PORT    I2S_NUM_0
#define SAMPLE_RATE 16000
#define DMA_COUNT   4
#define DMA_LEN     128   // samples per DMA buffer

// MAX30102 rolling buffer length required by SparkFun SpO2 algorithm
#define MAX30102_BUF 100

// Sensor objects
MAX30105 particleSensor;
MPU6050  imu;
bool     hasMAX = false;
bool     hasIMU = false;

// MAX30102 rolling sample buffers
uint32_t irBuf [MAX30102_BUF];
uint32_t redBuf[MAX30102_BUF];
int32_t  spo2Val   = 0;  int8_t spo2Valid  = 0;
int32_t  hrPPGVal  = 0;  int8_t hrPPGValid = 0;

BLECharacteristic *dataCharacteristic;
bool deviceConnected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) {
    deviceConnected = true;
  }
  void onDisconnect(BLEServer* server) {
    deviceConnected = false;
    BLEDevice::startAdvertising();
  }
};

// I2S initialisation for stereo SPH0645LM4H-B
void i2s_init() {
  i2s_config_t cfg = {};
  cfg.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate          = SAMPLE_RATE;
  cfg.bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT;  // SPH0645 uses 32-bit frame
  cfg.channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT;  // L = chest, R = ambient
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count        = DMA_COUNT;
  cfg.dma_buf_len          = DMA_LEN;
  cfg.use_apll             = true;
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);

  i2s_pin_config_t pins = {};
  pins.bck_io_num   = I2S_SCK;
  pins.ws_io_num    = I2S_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num  = I2S_DIN;
  i2s_set_pin(I2S_PORT, &pins);
}

// R-peak detector for ECG heart rate (derivative + adaptive threshold)
// Call at 250 Hz. Returns current HR estimate in bpm.
float detectHR(int raw) {
  static int   prev  = 0;
  static float thr   = 400.0f;
  static float hr    = 0.0f;
  static uint32_t last = 0;

  int d = raw - prev;
  prev  = raw;
  uint32_t now = millis();

  if (d > thr && (now - last) > 300) {
    uint32_t rr = now - last;
    if (rr > 300 && rr < 2000) hr = 60000.0f / rr;
    last = now;
    thr  = d * 0.6f;
  } else {
    thr = thr * 0.99f + fabsf((float)d) * 0.004f;
    if (thr < 150.0f) thr = 150.0f;
  }
  return hr;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);
  Wire.begin(I2C_SDA, I2C_SCL);

  i2s_init();

  // MAX30102 — prime rolling buffer for first SpO2 calculation
  hasMAX = particleSensor.begin(Wire, I2C_SPEED_FAST);
  if (hasMAX) {
    particleSensor.setup(60, 4, 2, 100, 411, 4096);
    for (int i = 0; i < MAX30102_BUF; i++) {
      while (!particleSensor.available()) particleSensor.check();
      redBuf[i] = particleSensor.getRed();
      irBuf[i]  = particleSensor.getIR();
      particleSensor.nextSample();
    }
    maxim_heart_rate_and_oxygen_saturation(irBuf, MAX30102_BUF, redBuf,
      &spo2Val, &spo2Valid, &hrPPGVal, &hrPPGValid);
  } else {
    Serial.println("[WARN] MAX30102 not found");
  }

  // MPU-6050
  imu.initialize();
  hasIMU = imu.testConnection();
  if (!hasIMU) Serial.println("[WARN] MPU-6050 not found");

  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);

  dataCharacteristic = service->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );

  dataCharacteristic->addDescriptor(new BLE2902());
  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->setMinPreferred(0x06);
  advertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("Smart Stethoscope BLE started.");
}

void loop() {
  static unsigned long lastSend  = 0;
  static unsigned long lastECG   = 0;
  static unsigned long lastSpO2  = 0;
  static int   sampleIndex = 0;
  static float ecgHR       = 0.0f;

  unsigned long now = millis();

  // Sample ECG at 250 Hz for R-peak HR detection
  if (now - lastECG >= 4) {
    lastECG = now;
    ecgHR = detectHR(analogRead(ECG_PIN));
  }

  // Update SpO2 every 2 s (slide rolling window by 25 new samples)
  if (hasMAX && (now - lastSpO2 >= 2000)) {
    lastSpO2 = now;
    memmove(redBuf, redBuf + 25, (MAX30102_BUF - 25) * sizeof(uint32_t));
    memmove(irBuf,  irBuf  + 25, (MAX30102_BUF - 25) * sizeof(uint32_t));
    int n = 0;
    unsigned long t0 = millis();
    while (n < 25 && millis() - t0 < 400) {
      particleSensor.check();
      if (particleSensor.available()) {
        redBuf[MAX30102_BUF - 25 + n] = particleSensor.getRed();
        irBuf [MAX30102_BUF - 25 + n] = particleSensor.getIR();
        particleSensor.nextSample();
        n++;
      }
    }
    if (n == 25)
      maxim_heart_rate_and_oxygen_saturation(irBuf, MAX30102_BUF, redBuf,
        &spo2Val, &spo2Valid, &hrPPGVal, &hrPPGValid);
  }

  // Send packet at ~20 Hz — same structure as original
  if (now - lastSend >= 50) {
    lastSend = now;

    // ADC reads — unchanged from original
    int ecgRaw      = analogRead(ECG_PIN);
    int pressureRaw = analogRead(PRESSURE_PIN);
    int piezo1Raw   = analogRead(PIEZO1_PIN);
    int piezo2Raw   = analogRead(PIEZO2_PIN);

    // Heart rate: use ECG R-peak result; fall back to PPG if ECG not yet locked
    int heartRate = (ecgHR > 0) ? (int)ecgHR : (int)hrPPGVal;

    // SpO2 from MAX30102; keep original fallback value if sensor not ready
    int spo2 = (hasMAX && spo2Valid && spo2Val > 0) ? (int)spo2Val : 97;

    // Respiration — placeholder until dedicated sensor is wired (Phase 2)
    int respiration = 16;

    // Temperature — placeholder until NTC/DS18B20 is wired (Phase 2)
    float temperature = 36.7;

    // IMU from MPU-6050; keep original fallback values if sensor absent
    float imuX = 0.02, imuY = 0.03, imuZ = 1.00;
    if (hasIMU) {
      int16_t ax, ay, az, gx, gy, gz;
      imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
      imuX = ax / 16384.0f;  // ±2g full-scale
      imuY = ay / 16384.0f;
      imuZ = az / 16384.0f;
    }

    // MEMS mic read via I2S — drains DMA buffer and takes the latest stereo pair
    // SPH0645: 24-bit audio in bits[31:8] of each 32-bit word; shift >> 16 for int16
    int32_t i2sBuf[DMA_LEN * 2];
    size_t  got = 0;
    i2s_read(I2S_PORT, i2sBuf, sizeof(i2sBuf), &got, 0);  // non-blocking
    int chestMic   = 2000;  // defaults if I2S not yet ready
    int ambientMic = 1500;
    if (got >= 8) {
      int n = (int)(got / (2 * sizeof(int32_t)));
      chestMic   = (int)(i2sBuf[(n - 1) * 2]     >> 16);
      ambientMic = (int)(i2sBuf[(n - 1) * 2 + 1] >> 16);
    }

    // Ambient noise subtraction before sending (from DSP pipeline spec)
    int heart_sound   = chestMic - (int)(0.8f * ambientMic);
    int ambient_sound = ambientMic;

    char payload[512];

    snprintf(payload, sizeof(payload),
      "{\"seq\":%d,\"ecg\":%d,\"heart_sound\":%d,\"ambient_sound\":%d,"
      "\"heart_rate\":%d,\"spo2\":%d,\"respiration\":%d,\"temperature\":%.2f,"
      "\"pressure\":%d,\"piezo1\":%d,\"piezo2\":%d,"
      "\"imu_x\":%.3f,\"imu_y\":%.3f,\"imu_z\":%.3f}",
      sampleIndex, ecgRaw, heart_sound, ambient_sound, heartRate, spo2, respiration,
      temperature, pressureRaw, piezo1Raw, piezo2Raw, imuX, imuY, imuZ
    );

    if (deviceConnected) {
      dataCharacteristic->setValue((uint8_t*)payload, strlen(payload));
      dataCharacteristic->notify();
    }

    Serial.println(payload);
    sampleIndex++;
  }
}
