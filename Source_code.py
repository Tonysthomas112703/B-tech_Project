import cv2
import numpy as np
import pyttsx3
import time
import RPi.GPIO as GPIO
import subprocess
import sys

# Load MobileNet SSD model and configuration
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Define classes for the COCO dataset
CLASSES = ["something", "aeroplane", "bicycle", "bird", "boat",
           "cell phone", "bus", "car", "cat", " chair", "cow", "table",
           "dog", "horse", "bike", "person", "plants", "sheep",
           "sofa", "train", "plane sufrace"]

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Open a connection to the webcam without specifying a backend
cap = cv2.VideoCapture(0)

# Focal length of your camera (in pixel units)
focal_length = 600  # Example value, replace with your camera's focal length

# Desired range in meter
desired_range = 1

# Wait for 50 milliseconds between frames
wait_time = 4

# GPIO pin number for shutdown
SHUTDOWN_PIN = 18

# Dictionary to store last spoken time for each detected object
last_spoken_time = {}

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SHUTDOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def shutdown():
    print("Shutdown sequence initiated...")
    subprocess.call(["sudo", "shutdown", "-h", "now"])
    GPIO.cleanup()
    sys.exit()

# Callback function for GPIO pin
def button_callback(channel):
    shutdown()

# Register button callback
GPIO.add_event_detect(SHUTDOWN_PIN, GPIO.FALLING, callback=button_callback, bouncetime=2000)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the image for MobileNet SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    # Initialize a list to store detected objects
    detected_objects = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by confidence threshold
        if confidence > 0.7:
            # Extract the index of the class label
            class_id = int(detections[0, 0, i, 1])

            # Ensure the class_id is within the range of CLASSES list
            if 0 <= class_id < len(CLASSES):
                # Get the class name
                class_name = CLASSES[class_id]

                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Calculate object's distance from the camera using depth information
                # Replace this calculation with your own method of obtaining depth information
                # Example calculation using focal length and object size in the image
                object_width = endX - startX
                distance = (desired_range * focal_length) / object_width
                print(distance)

                # Check if the object is within the desired range
                if (distance > desired_range) and (distance <= desired_range + 0.7):
                    # Check if the object has been spoken in the last 10 seconds
                    current_time = time.time()
                    last_spoken = last_spoken_time.get(class_name, 0)
                    if current_time - last_spoken > 6:
                        # Speak out the detected object
                        engine.say(f"{class_name} detected")
                        engine.runAndWait()

                        # Update last spoken time for the object
                        last_spoken_time[class_name] = current_time

                        # Add detected object to the list
                        detected_objects.append(class_name)

    # Break the loop when 'q' key is pressed
    # if cv2.waitKey(wait_time) & 0xFF == ord('q'):
    #     break

# Release the webcam
cap.release()

# Cleanup GPIO
GPIO.cleanup()