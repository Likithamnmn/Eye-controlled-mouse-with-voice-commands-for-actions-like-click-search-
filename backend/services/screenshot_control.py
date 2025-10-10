import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

import cv2
import numpy as np
import time
import pyautogui
from datetime import datetime

print("Starting screenshot control script...")

# Create screenshots folder if it doesn't exist
screenshots_folder = "screenshots_captured"
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)
    print(f"✓ Created folder: {screenshots_folder}")
else:
    print(f"✓ Using existing folder: {screenshots_folder}")

try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    print("✓ MediaPipe hands initialized successfully")
    
except Exception as e:
    print(f"✗ Error importing MediaPipe: {e}")
    exit(1)

# Webcam dimensions
wCam, hCam = 640, 480

print("Initializing webcam...")
# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Error: Cannot open webcam")
    exit(1)

cap.set(3, wCam)
cap.set(4, hCam)
print("✓ Webcam initialized successfully")

# Screenshot variables
last_screenshot_time = 0
screenshot_cooldown = 2  # 2 seconds between screenshots
pTime = 0

def fingers_up(lm_list):
    """
    Detect which fingers are up
    Returns list of 1s and 0s for [thumb, index, middle, ring, pinky]
    """
    fingers = []
    
    # Thumb (compare x coordinates due to hand orientation)
    if lm_list[4][1] > lm_list[3][1]:  # Thumb tip > thumb joint
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other four fingers (compare y coordinates)
    for id in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky tips
        if lm_list[id][2] < lm_list[id-2][2]:  # Tip < joint
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def take_screenshot():
    """Take a screenshot and save it in the screenshots_captured folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(screenshots_folder, filename)
    
    try:
        # Take screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        print(f"✓ Screenshot saved: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error taking screenshot: {e}")
        return False

print("Starting gesture detection...")
print("Gestures:")
print("- Open Palm (all fingers up) → Take Screenshot")
print("- Closed Fist (no fingers up) → Take Screenshot")
print("- Peace Sign (index + middle up) → Take Screenshot")
print("Press 'Esc' key to exit")

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        print("Frame captured")

        img = cv2.flip(img, 1)  # Flip for mirror effect
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = hands.process(img_rgb)
        
        gesture_detected = False
        gesture_name = ""
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                lm_list = []
                h, w, c = img.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                
                if len(lm_list) >= 21:  # Full hand landmarks
                    fingers = fingers_up(lm_list)
                    total_fingers = sum(fingers)
                    
                    # Gesture detection
                    current_time = time.time()
                    
                    # Open Palm (all 5 fingers up)
                    if total_fingers == 5:
                        gesture_detected = True
                        gesture_name = "Open Palm"
                    
                    # Closed Fist (no fingers up)
                    elif total_fingers == 0:
                        gesture_detected = True
                        gesture_name = "Closed Fist"
                    
                    # Peace Sign (index + middle fingers up)
                    elif fingers == [0, 1, 1, 0, 0]:
                        gesture_detected = True
                        gesture_name = "Peace Sign"
                    
                    # Take screenshot if gesture detected and cooldown passed
                    if gesture_detected and (current_time - last_screenshot_time) > screenshot_cooldown:
                        if take_screenshot():
                            last_screenshot_time = current_time
                            # Visual feedback
                            cv2.rectangle(img, (50, 50), (300, 100), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, "Screenshot Taken!", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Display current gesture
                    if gesture_detected:
                        cv2.putText(img, f"Gesture: {gesture_name}", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img, f"Fingers: {total_fingers}", (10, 180), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Display finger status
                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    for i, (finger, status) in enumerate(zip(finger_names, fingers)):
                        color = (0, 255, 0) if status else (0, 0, 255)
                        cv2.putText(img, f"{finger}: {'Up' if status else 'Down'}", 
                                  (400, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show cooldown timer
        remaining_cooldown = max(0, screenshot_cooldown - (time.time() - last_screenshot_time))
        if remaining_cooldown > 0:
            cv2.putText(img, f"Cooldown: {remaining_cooldown:.1f}s", (10, 400), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Show output
        cv2.imshow("Screenshot Control", img)

        # Exit on 'Esc' key
        if cv2.waitKey(1) == 27:
            break

except Exception as e:
    print(f"✗ Error in main loop: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Cleanup complete")