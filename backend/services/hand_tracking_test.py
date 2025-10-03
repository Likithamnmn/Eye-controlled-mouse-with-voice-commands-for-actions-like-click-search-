import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

import cv2
import numpy as np
import time
import math

print("Starting simple hand tracking test...")

try:
    import HandTrackingModule as htm
    print("✓ HandTrackingModule imported successfully")
except Exception as e:
    print(f"✗ Error importing HandTrackingModule: {e}")
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

print("Starting hand tracking...")
print("Move your thumb and index finger close together and apart")
print("Press 'Esc' key to exit")

pTime = 0

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam")
            break

        img = cv2.flip(img, 1)  # Flip for mirror effect
        
        # Use HandTrackingModule's FindHands function
        img, lm_list = htm.FindHands(img)
        
        if lm_list:
            # Get landmark positions for thumb tip (4) and index finger tip (8)
            if len(lm_list) >= 9:  # Make sure we have enough landmarks
                x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
                x2, y2 = lm_list[8][1], lm_list[8][2]  # Index tip
                
                # Calculate distance between thumb and index finger
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
                
                # Display distance
                cv2.putText(img, f'Distance: {int(length)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                
                # Simulate volume percentage
                volPer = np.interp(length, [50, 300], [0, 100])
                cv2.putText(img, f'Volume: {int(volPer)}%', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Show output
        cv2.imshow("Hand Tracking Test", img)

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