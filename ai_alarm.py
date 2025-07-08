import cv2
import numpy as np
# Consider using tflite_runtime for deployment on Raspberry Pi
# try:
#     import tflite_runtime.interpreter as tflite_interpreter
# except ImportError:
import tensorflow as tf # Fallback for development/testing
from gpiozero import MotionSensor, LED, Buzzer
from time import sleep
import sys # For graceful exit

# --- Configuration ---
PIR_PIN = 17
LED_PIN = 27
BUZZER_PIN = 22
MODEL_PATH = 'mobilenet_v1.tflite'
LABELS_PATH = 'labels.txt'
CONFIDENCE_THRESHOLD = 0.6 # Only trigger if confidence is above this
ALARM_DURATION_SEC = 2
COOLDOWN_AFTER_ALARM_SEC = 5 # Prevent immediate re-triggering

# --- GPIO Setup ---
try:
    pir = MotionSensor(PIR_PIN)
    led = LED(LED_PIN)
    buzzer = Buzzer(BUZZER_PIN)
    print("GPIO devices initialized.")
except Exception as e:
    print(f"Error initializing GPIO devices: {e}")
    print("Ensure you are running on a Raspberry Pi and gpiozero is installed.")
    sys.exit(1) # Exit if GPIO setup fails

# --- Load Labels ---
try:
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(labels)} labels from {LABELS_PATH}")
except FileNotFoundError:
    print(f"Error: labels.txt not found at {LABELS_PATH}")
    sys.exit(1)

# --- Load TensorFlow Lite Model ---
try:
    tflite_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    print(f"Loaded TensorFlow Lite model: {MODEL_PATH}")
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")

    # Check input tensor type and shape
    if input_details[0]['dtype'] != np.float32:
        print(f"Warning: Input tensor expected float32, but found {input_details[0]['dtype']}. Conversion will be applied.")
    if input_details[0]['shape'][1:3] != (224, 224):
        print(f"Warning: Model expects input shape {input_details[0]['shape']}, but resizing to (224, 224).")

except Exception as e:
    print(f"Error loading TensorFlow Lite model: {e}")
    sys.exit(1)

# --- Camera Setup ---
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)
print("Camera initialized.")


print("AI Alarm Active — waiting for motion...")

try:
    while True:
        pir.wait_for_motion()
        print("Motion detected!")

        # Grab frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            sleep(0.1) # Small delay before retrying
            continue

        # Preprocess for MobileNet
        # Ensure the image is resized to what the model expects
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1:3]

        img = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(img, axis=0)

        # Normalize pixel values to [0, 1] if the model expects it (common for MobileNet)
        # Check input_details[0]['quantization'] for min/max if quantized, otherwise assume float32 input 0-255 or 0-1
        if input_details[0]['dtype'] == np.float32:
            input_data = np.float32(input_data) / 255.0 # Normalize to 0-1 range
        else:
            input_data = np.uint8(input_data) # Ensure it's the correct type if quantized

        tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
        tflite_interpreter.invoke()

        output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
        probabilities = tf.nn.softmax(output_data).numpy().flatten() # Get probabilities and flatten

        top_result_index = np.argmax(probabilities)
        top_confidence = probabilities[top_result_index]
        top_label = labels[top_result_index]

        print(f"Detected: {top_label} with confidence: {top_confidence:.2f}")

        # If 'person' found with sufficient confidence, trigger alarm
        # This assumes 'person' is an exact label in labels.txt
        if 'person' in top_label.lower() and top_confidence >= CONFIDENCE_THRESHOLD:
            print("PERSON DETECTED! ALARM ACTIVATED!")
            led.on()
            buzzer.on()
            sleep(ALARM_DURATION_SEC)
            led.off()
            buzzer.off()
            print("Alarm deactivated. Waiting for no motion...")
            # After alarm, wait for motion to clear and then a cooldown
            pir.wait_for_no_motion()
            sleep(COOLDOWN_AFTER_ALARM_SEC) # Cooldown period
        else:
            print("No person detected or confidence too low. Waiting for no motion...")
            pir.wait_for_no_motion() # Wait for motion to clear before re-arming PIR

except KeyboardInterrupt:
    print("\nExiting AI Alarm system.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Ensure camera and GPIO resources are released
    if 'camera' in locals() and camera.isOpened():
        camera.release()
        print("Camera released.")
    if 'led' in locals():
        led.off()
    if 'buzzer' in locals():
        buzzer.off()
    print("GPIO devices reset.")
    sys.exit(0) # Clean exit
