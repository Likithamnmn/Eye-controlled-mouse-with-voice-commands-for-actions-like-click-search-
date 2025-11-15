from flask import Blueprint, jsonify
import threading
from services import pure_eye_calibrator
from utils.feature_flags import is_enabled
import cv2,time
from services import volume_control

# Force-release any previous camera handle

bp = Blueprint("control", __name__)

# Keep controller refs globally for stop commands
controllers = {
    "voice": None,
    "eye": None,
}

# Thread control flags
eye_tracker_instance = None
eye_tracker_stop_event = None
eye_tracker_thread = None

# --- FEATURES ---
@bp.route("/features", methods=["GET"])
def features():
    from utils.feature_flags import FEATURES
    return jsonify({"features": FEATURES})


# -------------------- EYE CONTROL --------------------
# Global flag to signal eye tracker to stop
eye_tracker_should_stop = False

def run_eye_control():
    from services import advanced_eye_tracker
    global controllers, eye_tracker_instance, eye_tracker_should_stop
    
    eye_tracker_should_stop = False  # Reset stop flag
    
    try:
        # Set global stop flag in the service module BEFORE calling
        # This allows the service to check it during execution
        if not hasattr(advanced_eye_tracker, 'should_stop'):
            advanced_eye_tracker.should_stop = False
        
        # Create and run the tracker - this is a blocking call
        # The tracker will run until stopped or interrupted
        # NOTE: We need to modify the service to check should_stop flag
        tracker = advanced_eye_tracker.create_advanced_eye_tracking_demo()
        eye_tracker_instance = tracker
        
    except KeyboardInterrupt:
        print("Eye tracking interrupted")
    except Exception as e:
        print(f"Eye tracking error: {e}")
    finally:
        controllers["eye"] = None
        eye_tracker_instance = None
        eye_tracker_should_stop = False

@bp.route("/eye/start", methods=["POST"])
def eye_start():
    global controllers, eye_tracker_instance, eye_tracker_stop_event, eye_tracker_should_stop, eye_tracker_thread
    if not is_enabled("eye_control"):
        return jsonify({"error": "Eye control disabled"}), 403
    
    # If already running, stop it first to ensure clean start
    if controllers["eye"] is not None or eye_tracker_instance is not None:
        eye_stop_internal()
        # Wait a moment for cleanup
        import time
        time.sleep(0.5)
    
    try:
        # Run calibration first (blocking) so the tracker picks up the new file
        try:
            print("🔧 Running calibration before starting eye tracker...")
            calib_file = pure_eye_calibrator.run_pure_eye_calibration()
            if not calib_file:
                controllers["eye"] = None
                return jsonify({"error": "Calibration failed or cancelled"}), 400
            print(f"🔎 Calibration saved: {calib_file}")
        except Exception as e:
            controllers["eye"] = None
            return jsonify({"error": f"Calibration error: {str(e)}"}), 500

        # Use threading (multiprocessing doesn't share globals well)
        eye_tracker_stop_event = threading.Event()
        eye_tracker_should_stop = False
        controllers["eye"] = "running"  # Mark as running

        # Create thread - global variables will be shared
        eye_tracker_thread = threading.Thread(target=run_eye_control, daemon=True)
        eye_tracker_thread.start()

        return jsonify({"status": "Eye control started", "eye_active": True})
    except Exception as e:
        controllers["eye"] = None
        eye_tracker_instance = None
        eye_tracker_stop_event = None
        eye_tracker_thread = None
        eye_tracker_should_stop = False
        return jsonify({"error": f"Failed to start eye control: {str(e)}"}), 500

def eye_stop_internal():
    """Internal function to stop eye tracking - can be called from start or stop"""
    global controllers, eye_tracker_instance, eye_tracker_stop_event, eye_tracker_should_stop, eye_tracker_thread
    
    print("🛑 Attempting to stop eye tracker...")
    
    # Set stop flag - this will stop the service loop
    print("🛑 Setting stop flag to stop eye tracker...")
    
    # Set global stop flag
    eye_tracker_should_stop = True
    
    # Signal the thread to stop
    if eye_tracker_stop_event:
        eye_tracker_stop_event.set()
        print("🛑 Eye tracker stop event set")
    
    # Try to set stop flag in service module - this is the KEY to stopping
    try:
        from services import advanced_eye_tracker
        # Set the global stop flag - the service loop should check this
        advanced_eye_tracker.should_stop = True
        print("🛑 Set service.should_stop = True")
        
        # Also try to access the camera directly if possible
        if hasattr(advanced_eye_tracker, 'cap') and advanced_eye_tracker.cap:
            try:
                advanced_eye_tracker.cap.release()
                print("🛑 Released camera via service.cap.release()")
            except:
                pass
        if hasattr(advanced_eye_tracker, "running"):
            advanced_eye_tracker.running = False
            print("🛑 Set service.running = False")
        if hasattr(advanced_eye_tracker, "stop_all"):
            advanced_eye_tracker.stop_all()
            print("🛑 Called service.stop_all()")
        if hasattr(advanced_eye_tracker, "stop_eye_tracking"):
            advanced_eye_tracker.stop_eye_tracking()
            print("🛑 Called service.stop_eye_tracking()")
        # Try to access and release camera directly if possible
        if hasattr(advanced_eye_tracker, "cap") and advanced_eye_tracker.cap:
            try:
                advanced_eye_tracker.cap.release()
                print("🛑 Released camera via service.cap.release()")
            except:
                pass
        if hasattr(advanced_eye_tracker, "camera") and advanced_eye_tracker.camera:
            try:
                advanced_eye_tracker.camera.release()
                print("🛑 Released camera via service.camera.release()")
            except:
                pass
    except Exception as e:
        print(f"Error calling service stop methods: {e}")
    
    # Try to stop the tracker instance
    if eye_tracker_instance is not None:
        try:
            if hasattr(eye_tracker_instance, "stop"):
                eye_tracker_instance.stop()
                print("🛑 Called tracker.stop()")
            if hasattr(eye_tracker_instance, "stop_eye_tracking"):
                eye_tracker_instance.stop_eye_tracking()
                print("🛑 Called tracker.stop_eye_tracking()")
            if hasattr(eye_tracker_instance, "cleanup"):
                eye_tracker_instance.cleanup()
                print("🛑 Called tracker.cleanup()")
            if hasattr(eye_tracker_instance, "release"):
                eye_tracker_instance.release()
                print("🛑 Called tracker.release()")
            if hasattr(eye_tracker_instance, "close"):
                eye_tracker_instance.close()
                print("🛑 Called tracker.close()")
            # Try to access camera directly from instance
            if hasattr(eye_tracker_instance, "cap"):
                try:
                    eye_tracker_instance.cap.release()
                    print("🛑 Released camera via tracker.cap.release()")
                except:
                    pass
        except Exception as e:
            print(f"Error stopping eye tracker instance: {e}")
    
    # Try to release OpenCV camera directly (if accessible globally)
    try:
        import cv2
        # OpenCV might have a global camera that needs releasing
        # This is a last resort attempt
        print("🛑 Attempted to access OpenCV directly")
    except:
        pass
    
    controllers["eye"] = None
    eye_tracker_instance = None
    if eye_tracker_stop_event:
        eye_tracker_stop_event.clear()
    eye_tracker_stop_event = None
    eye_tracker_thread = None
    eye_tracker_should_stop = False
    print("🛑 Cleared all eye tracker references")

@bp.route("/eye/stop", methods=["POST"])
def eye_stop():
    try:
        eye_stop_internal()
        print("✅ Eye control stopped successfully")
        return jsonify({"status": "Eye control stopped", "eye_active": False})
    except Exception as e:
        print(f"Error in eye_stop: {e}")
        # Clear everything even on error
        controllers["eye"] = None
        eye_tracker_instance = None
        eye_tracker_stop_event = None
        return jsonify({"status": "Eye control stopped", "eye_active": False})


@bp.route("/eye/calibrate", methods=["POST"])
def eye_calibrate():
    if not is_enabled("eye_control"):
        return jsonify({"error": "Eye control disabled"}), 403
    threading.Thread(target=pure_eye_calibrator.run_pure_eye_calibration, daemon=True).start()
    return jsonify({"status": "Eye calibration started"})


# -------------------- HAND GESTURE CONTROL --------------------
# Module-level functions to avoid pickling issues
def run_volume_control():
    """Run volume control (pinch gesture) in a separate thread"""
    from services import volume_screenshot_core
    print("🎥 Starting volume control (pinch gesture) - camera should initialize...")
    try:
        volume_screenshot_core.volume_screenshot_core_loop(True, False)
    except Exception as e:
        print(f"❌ Volume control error: {e}")
        import traceback
        traceback.print_exc()

screenshot_running = True
cap = None
hands = None

def run_screenshot_control():
    global screenshot_running
    last_time = 0
    cooldown = 2
    while screenshot_running:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = [[i, int(lm.x*640), int(lm.y*480)]
                           for i, lm in enumerate(hand_landmarks.landmark)]
                if len(lm_list) == 21:
                    fingers = fingers_up(lm_list)
                    if sum(fingers) == 5 and time.time() - last_time > cooldown:
                        take_screenshot()
                        last_time = time.time()
        # Optional: small sleep to reduce CPU usage
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Screenshot control stopped")



# Track running threads
gesture_threads = {
    "volume": None,
    "screenshot": None,
}



@bp.route("/volume/control/start", methods=["POST"])
def volume_start():
    global gesture_threads

    # Stop existing volume thread if running
    if gesture_threads["volume"] and gesture_threads["volume"].is_alive():
        volume_stop()
        time.sleep(0.1)

    # Start new volume thread
    gesture_threads["volume"] = threading.Thread(target=volume_control.start_volume_control, daemon=True)
    gesture_threads["volume"].start()
    return jsonify({"status": "Volume control started - camera initializing"})

@bp.route("/volume/control/stop", methods=["POST"])
def volume_stop():
    global gesture_threads

    # Stop volume control
    volume_control.stop_volume_control()

    # Wait briefly to ensure resources are released
    time.sleep(0.1)
    gesture_threads["volume"] = None
    return jsonify({"status": "Volume control stopped"})

@bp.route("/screenshot/control/start", methods=["POST"])
def screenshot_start():
    global gesture_threads
    import time
    from services.screenshot_control import run_screenshot_control

    # Stop volume if running
    if gesture_threads["volume"] and gesture_threads["volume"].is_alive():
        from services.volume_control import stop_volume_control
        stop_volume_control()
        time.sleep(0.2)

    # Stop existing screenshot thread
    if gesture_threads["screenshot"] and gesture_threads["screenshot"].is_alive():
        screenshot_stop()
        time.sleep(0.2)

    print("🖐️ Starting palm gesture (screenshot) control...")
    gesture_threads["screenshot"] = threading.Thread(target=run_screenshot_control, daemon=True)
    gesture_threads["screenshot"].start()
    return jsonify({"status": "Screenshot control started - camera initializing"})


@bp.route("/screenshot/control/stop", methods=["POST"])
def screenshot_stop():
    global gesture_threads
    from services.screenshot_control import stop_screenshot_control

    try:
        stop_screenshot_control()
        print("🛑 Screenshot control stopped")
    except Exception as e:
        print(f"Error stopping screenshot: {e}")

    gesture_threads["screenshot"] = None
    return jsonify({"status": "Screenshot control stopped"})



# -------------------- VOICE CONTROL --------------------
def run_voice_controller(voice_controller):
    """Run voice controller in a separate thread - defined at module level to avoid pickling issues"""
    print(">>> Voice controller thread started")
    try:
        voice_controller.run()
    except Exception as e:
        print(f"Voice controller error: {e}")
    finally:
        print(">>> Voice controller thread ended")
        # Clear controller reference when thread ends
        global controllers
        controllers["voice"] = None

@bp.route("/voice/start", methods=["POST"])
def voice_start():
    from services.voice_browser_control import VoiceBrowserController
    global controllers

    # If already running, stop it first to ensure clean start
    if controllers["voice"] is not None:
        voice_stop_internal()
        # Wait a moment for cleanup
        import time
        time.sleep(0.3)

    try:
        vc = VoiceBrowserController()
        controllers["voice"] = vc

        # Use threading for voice control - pass the controller as argument
        threading.Thread(target=run_voice_controller, args=(vc,), daemon=True).start()
        return jsonify({"status": "Voice control started", "voice_active": True})
    except Exception as e:
        controllers["voice"] = None
        return jsonify({"error": f"Failed to start voice control: {str(e)}"}), 500


def voice_stop_internal():
    """Internal function to stop voice control"""
    global controllers
    vc = controllers.get("voice")
    
    print("🛑 Attempting to stop voice controller...")
    
    if vc is None:
        controllers["voice"] = None
        return
    
    # Try multiple stop methods
    try:
        if hasattr(vc, "stop_all"):
            vc.stop_all()
            print("🛑 Called vc.stop_all()")
        if hasattr(vc, "stop"):
            vc.stop()
            print("🛑 Called vc.stop()")
        if hasattr(vc, "stop_listening"):
            vc.stop_listening()
            print("🛑 Called vc.stop_listening()")
        if hasattr(vc, "cleanup"):
            vc.cleanup()
            print("🛑 Called vc.cleanup()")
        if hasattr(vc, "close"):
            vc.close()
            print("🛑 Called vc.close()")
    except Exception as e:
        print(f"Error stopping voice controller methods: {e}")
    
    # Try to set stop flags if they exist
    try:
        if hasattr(vc, "should_stop"):
            vc.should_stop = True
            print("🛑 Set vc.should_stop = True")
        if hasattr(vc, "running"):
            vc.running = False
            print("🛑 Set vc.running = False")
        if hasattr(vc, "is_listening"):
            vc.is_listening = False
            print("🛑 Set vc.is_listening = False")
    except Exception as e:
        print(f"Error setting voice controller flags: {e}")
    
    # Clear controller reference
    controllers["voice"] = None
    print("✅ Voice controller reference cleared")

@bp.route("/voice/stop", methods=["POST"])
def voice_stop():
    try:
        voice_stop_internal()
        # Small delay to allow stop to propagate
        import time
        time.sleep(0.2)
        print("✅ Voice control stopped successfully")
        return jsonify({"status": "Voice control stopped", "voice_active": False})
    except Exception as e:
        print(f"Error in voice_stop: {e}")
        # Clear controller even on error
        controllers["voice"] = None
        return jsonify({"status": "Voice control stopped", "voice_active": False})


# -------------------- LEGACY COMPATIBILITY ROUTES --------------------
@bp.route("/start_screenshot_control", methods=["POST"])
def legacy_start_screenshot_control():
    """Legacy route for /control/start_screenshot_control"""
    return screenshot_start()

@bp.route("/stop_volume_screenshot", methods=["POST"])
def legacy_stop_volume_screenshot():
    """Legacy route for /control/stop_volume_screenshot"""
    return both_stop()

@bp.route("/eye/move", methods=["POST"])
def eye_move():
    """Placeholder for eye movement tracking"""
    return jsonify({"status": "Eye movement tracking not yet implemented"}), 200

