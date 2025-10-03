import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

import cv2
import numpy as np
import time
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

print("Starting volume control script...")

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

print("Initializing audio control...")
try:
    # Initialize audio control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    # Get volume range
    volRange = volume.GetVolumeRange()
    minVol, maxVol = volRange[0], volRange[1]
    print(f"✓ Audio control initialized. Volume range: {minVol} to {maxVol}")
except Exception as e:
    print(f"✗ Error initializing audio control: {e}")
    exit(1)

print("Initializing hand detector...")
try:
    # Hand detection is already initialized above with MediaPipe
    print("✓ Hand detector initialized successfully")
except Exception as e:
    print(f"✗ Error initializing hand detector: {e}")
    exit(1)

# Volume bar & percentage display
volBar, volPer = 400, 0

pTime = 0  # Previous time for FPS calculation

print("Starting main loop...")
print("Press 'Esc' key to exit")

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam")
            break

        img = cv2.flip(img, 1)  # Flip for mirror effect
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = hands.process(img_rgb)
        
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
                
                if len(lm_list) >= 9:  # Make sure we have enough landmarks
                    # Thumb tip (4) and Index finger tip (8)
                    x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
                    x2, y2 = lm_list[8][1], lm_list[8][2]  # Index tip
                    
                    # Calculate distance
                    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    
                    # Draw circles on fingertips
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    # Draw connection point
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    
                    # Change color if fingers are close
                    if length < 50:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                    
                    # Map length to volume range
                    vol = np.interp(length, [50, 300], [minVol, maxVol])
                    volBar = np.interp(length, [50, 300], [400, 150])
                    volPer = np.interp(length, [50, 300], [0, 100])
                    
                    # Draw volume bar
                    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                    
                    # Set system volume
                    volume.SetMasterVolumeLevel(vol, None)

        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Show output
        cv2.imshow("Volume Control", img)

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