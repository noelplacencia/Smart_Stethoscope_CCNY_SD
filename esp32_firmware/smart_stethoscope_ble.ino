/*
  Smart Stethoscope ESP32-S3 BLE Firmware
  Sends sensor data from ESP32-S3 to Raspberry Pi by BLE.

  Current version:
  - Uses real analog reads for AD8232 ECG, pressure, and piezo sensors.
  - Uses simulated values for SpO2, heart rate, respiration, IMU, temperature,
    and microphone envelope until full sensor libraries are integrated.

  BLE Device Name: SmartStethoscope_ESP32
*/

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define DEVICE_NAME "SmartStethoscope_ESP32"

#define SERVICE_UUID        "7b4d0001-8a7b-4d2b-9b41-000000000001"
#define CHARACTERISTIC_UUID "7b4d0002-8a7b-4d2b-9b41-000000000001"

// Change these pins according to final wiring
#define ECG_PIN       4
#define PRESSURE_PIN  5
#define PIEZO1_PIN    6
#define PIEZO2_PIN    7

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

float fakeWave(float t, float freq, float amp, float offset) {
  return offset + amp * sin(2.0 * PI * freq * t);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);

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
  static unsigned long lastSend = 0;
  static int sampleIndex = 0;

  unsigned long now = millis();

  // Sends about 20 packets per second
  if (now - lastSend >= 50) {
    lastSend = now;

    float t = now / 1000.0;

    int ecgRaw = analogRead(ECG_PIN);
    int pressureRaw = analogRead(PRESSURE_PIN);
    int piezo1Raw = analogRead(PIEZO1_PIN);
    int piezo2Raw = analogRead(PIEZO2_PIN);

    int heartRate = 72 + (int)(5 * sin(t * 0.5));
    int spo2 = 97 + (int)(1 * sin(t * 0.2));
    int respiration = 16 + (int)(2 * sin(t * 0.3));
    float temperature = 36.7 + 0.2 * sin(t * 0.1);

    float imuX = 0.02 * sin(t);
    float imuY = 0.03 * cos(t);
    float imuZ = 1.00;

    int chestMic = (int)fakeWave(t, 2.0, 400, 2000) + random(-80, 80);
    int ambientMic = random(1200, 1800);

    char payload[512];

    snprintf(payload, sizeof(payload),
      "{\"seq\":%d,\"ecg\":%d,\"heart_sound\":%d,\"ambient_sound\":%d,"
      "\"heart_rate\":%d,\"spo2\":%d,\"respiration\":%d,\"temperature\":%.2f,"
      "\"pressure\":%d,\"piezo1\":%d,\"piezo2\":%d,"
      "\"imu_x\":%.3f,\"imu_y\":%.3f,\"imu_z\":%.3f}",
      sampleIndex, ecgRaw, chestMic, ambientMic, heartRate, spo2, respiration,
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
