import cv2
import mediapipe as mp
import math
import numpy as np
import threading
import time
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Globals
volume_running = False
cap = None
hands = None
volume_interface = None

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize audio once
def init_audio():
    global volume_interface
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_interface = interface.QueryInterface(IAudioEndpointVolume)
    return volume_interface

# Start volume control
def start_volume_control():
    global volume_running, cap, hands, volume_interface
    if volume_running:
        return
    volume_running = True

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)

    if volume_interface is None:
        init_audio()

    threading.Thread(target=run_volume_control, daemon=True).start()

# Stop volume control
def stop_volume_control():
    global volume_running
    volume_running = False
    time.sleep(0.05)  # let thread exit

# Main loop
def run_volume_control():
    global volume_running, cap, hands, volume_interface
    pTime = 0
    volBar, volPer = 400, 0
    minVol, maxVol = volume_interface.GetVolumeRange()[0], volume_interface.GetVolumeRange()[1]

    while volume_running:
        if cap is None or not cap.isOpened():
            time.sleep(0.01)
            continue

        ret, img = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm_list = [[i, int(lm.x*640), int(lm.y*480)] 
                           for i, lm in enumerate(hand_landmarks.landmark)]

                if len(lm_list) >= 9:
                    x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb tip
                    x2, y2 = lm_list[8][1], lm_list[8][2]  # Index tip

                    length = math.hypot(x2 - x1, y2 - y1)
                    vol = np.interp(length, [50, 300], [minVol, maxVol])
                    volBar = np.interp(length, [50, 300], [400, 150])
                    volPer = np.interp(length, [50, 300], [0, 100])

                    volume_interface.SetMasterVolumeLevel(vol, None)

        # Optional: sleep to reduce CPU usage
        time.sleep(0.01)

    # Cleanup
    if cap:
        cap.release()
        cap = None
    if hands:
        hands.close()
        hands = None
    print("✓ Volume control stopped")
