import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import cv2
import numpy as np
import time
import math
import pyautogui
from datetime import datetime
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

try:
    import mediapipe as mp
except Exception as e:
    print(f"✗ MediaPipe import error: {e}")
    exit(1)

# --- Settings ---
wCam, hCam = 640, 480
screenshots_folder = "screenshots_captured"
os.makedirs(screenshots_folder, exist_ok=True)

screenshot_cooldown = 2  # seconds
volBar, volPer = 400, 0
pTime = 0

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- Audio Control ---
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
minVol, maxVol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]

# --- Helpers ---
def fingers_up(lm_list):
    fingers = []
    # Thumb
    fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)
    # Other fingers
    for id in [8, 12, 16, 20]:
        fingers.append(1 if lm_list[id][2] < lm_list[id-2][2] else 0)
    return fingers

def take_screenshot():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(screenshots_folder, f"screenshot_{timestamp}.png")
    try:
        pyautogui.screenshot().save(filepath)
        print(f"✓ Screenshot saved: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Screenshot error: {e}")
        return False

# --- Core Loop ---
def volume_screenshot_core_loop(run_volume=True, run_screenshot=True):
    global pTime, volBar, volPer
    last_screenshot_time = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("✗ Failed to read frame")
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm_list = []
                    h, w, c = img.shape
                    for id, lm in enumerate(hand_landmarks.landmark):
                        lm_list.append([id, int(lm.x*w), int(lm.y*h)])

                    if len(lm_list) >= 21:
                        fingers = fingers_up(lm_list)
                        total_fingers = sum(fingers)
                        cTime = time.time()

                        # --- Volume Control ---
                        if run_volume and len(lm_list) >= 9:
                            x1, y1 = lm_list[4][1], lm_list[4][2]
                            x2, y2 = lm_list[8][1], lm_list[8][2]
                            length = math.hypot(x2-x1, y2-y1)

                            vol = np.interp(length, [50, 300], [minVol, maxVol])
                            volBar = np.interp(length, [50, 300], [400, 150])
                            volPer = np.interp(length, [50, 300], [0, 100])
                            volume.SetMasterVolumeLevel(vol, None)

                            # Draw
                            cv2.rectangle(img, (50, 150), (85, 400), (255,0,0), 3)
                            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255,0,0), cv2.FILLED)
                            cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
                            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
                            cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
                            cv2.circle(img, (x2,y2), 10, (255,0,255), cv2.FILLED)
                            cx, cy = (x1+x2)//2, (y1+y2)//2
                            cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
                            if length < 50: cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)

                        # --- Screenshot Control ---
                        if run_screenshot:
                            gesture_detected = False
                            if total_fingers == 5: gesture_detected = True  # Open Palm
                            elif total_fingers == 0: gesture_detected = True  # Fist
                            elif fingers == [0,1,1,0,0]: gesture_detected = True  # Peace

                            if gesture_detected and (cTime - last_screenshot_time) >  screenshot_cooldown:
                                if take_screenshot(): last_screenshot_time = cTime
                                cv2.putText(img, "Screenshot Taken!", (60,80), cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime != 0 else 0
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

            cv2.imshow("Volume + Screenshot Control", img)
            if cv2.waitKey(1) == 27: break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")

# --- Run as script ---
if __name__ == "__main__":
    volume_screenshot_core_loop(run_volume=True, run_screenshot=True)
