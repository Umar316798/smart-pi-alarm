# Smart Pi Alarm

Smart Pi Alarm is a Raspberry Pi–powered security system that detects motion using a PIR sensor, captures an image, runs AI-based object detection using a TFLite model, and sounds an alarm when a person is detected.

### 🔧 Hardware Used
- Raspberry Pi 3 Model B+
- Raspberry Pi Camera Module V2
- HC-SR501 PIR Motion Sensor
- LED + Resistor
- Buzzer
- Breadboard + jumper wires
- MicroSD Card (64GB)
- Power Supply (5V, 2.5A+)

### 🧠 Features
- Motion detection using PIR sensor
- Image classification using TensorFlow Lite MobileNet
- LED and buzzer alarm for "person" detection
- Text-to-speech support using `espeak` (optional)

### 📦 Installation

#### On the Raspberry Pi:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install espeak python3-pip
pip3 install opencv-python tflite-runtime numpy
