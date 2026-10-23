import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import urllib.request
import os
import threading
from pynput import keyboard

# Disable safety failsafe
pyautogui.FAILSAFE = False

# Ensure model file exists
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe hand model...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)

# Initialize MediaPipe Tasks Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path=model_path,
    delegate=python.BaseOptions.Delegate.CPU
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Setup Webcam with DirectShow for fast Windows initialization
cam_w, cam_h = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
cap.set(cv2.CAP_PROP_FPS, 30)

# Thread-safe frame container for zero-lag capture
class WebcamStream:
    def __init__(self, stream):
        self.stream = stream
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Start background video thread
vs = WebcamStream(cap).start()

# Smoothing variables for buttery-smooth cursor glide
smoothening = 4.0 
curr_x, curr_y = 0.0, 0.0
pad_x, pad_y = 100, 80

# --- Bluetooth Neckband Background Listener ---
def on_press(key):
    try:
        if key == keyboard.Key.media_play_pause:
            pyautogui.click()
            print("-> Left Clicked via Neckband!")
        elif key == keyboard.Key.media_next:
            pyautogui.click(button='right')
            print("-> Right Clicked via Neckband!")
        elif key == keyboard.Key.media_previous:
            pyautogui.doubleClick()
            print("-> Double Clicked via Neckband!")
    except AttributeError:
        pass

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Run the neckband listener concurrently in a background daemon thread
click_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
click_thread.start()

print("==================================================")
print("  Unified Studio Hand Mouse & Neckband Active!")
print("==================================================")
print("• Zero-lag threaded hand skeleton tracking (Cursor)")
print("• Bluetooth neckband clicker active (Clicks)")
print("• Press 'q' on the video window to exit")

while True:
    frame = vs.read()
    if frame is None:
        continue

    # Flip horizontally for natural mirror movement
    frame = cv2.flip(frame, 1)
    
    # Convert OpenCV frame to MediaPipe Image format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process hand landmarks
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, _ = frame.shape
            
            # Draw skeleton joints
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                
            # Extract Index Fingertip (Landmark index 8)
            index_tip = hand_landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            
            # Map coordinates with padding
            target_x = np.interp(cx, (pad_x, w - pad_x), (0, screen_w))
            target_y = np.interp(cy, (pad_y, h - pad_y), (0, screen_h))
            
            # Apply EMA filtering for fluid glide
            curr_x = curr_x + (target_x - curr_x) / smoothening
            curr_y = curr_y + (target_y - curr_y) / smoothening
            
            # Move cursor
            pyautogui.moveTo(int(curr_x), int(curr_y))

    cv2.imshow("Unified Studio Hand Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cap.release()
cv2.destroyAllWindows()