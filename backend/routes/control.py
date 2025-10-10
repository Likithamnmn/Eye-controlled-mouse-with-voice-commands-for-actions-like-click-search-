from flask import Blueprint, jsonify
from services import pure_eye_calibrator
from utils.feature_flags import is_enabled
import threading

bp = Blueprint("control", __name__)

# --- FEATURES ---
@bp.route("/features", methods=["GET"])
def features():
    from utils.feature_flags import FEATURES
    return jsonify({"features": FEATURES})

# --- ADVANCED EYE CONTROL ---
def eye_tracking_thread():
    from services import advanced_eye_tracker
    advanced_eye_tracker.create_advanced_eye_tracking_demo()

@bp.route("/eye/move", methods=["POST"])
def eye_move():
    if not is_enabled("eye_control"):
        return jsonify({"error": "Eye control disabled"}), 403
    threading.Thread(target=eye_tracking_thread, daemon=True).start()
    return jsonify({"status": "Eye tracking started"})

@bp.route("/eye/calibrate", methods=["POST"])
def eye_calibrate():
    if not is_enabled("eye_control"):
        return jsonify({"error": "Eye control disabled"}), 403
    threading.Thread(target=pure_eye_calibrator.run_pure_eye_calibration, daemon=True).start()
    return jsonify({"status": "Eye calibration started"})

# --- VOLUME CONTROL ---
@bp.route("/start_volume_control", methods=["POST"])
def start_volume_control():
    def run_volume():
        from services import volume_screenshot_core
        volume_screenshot_core.volume_screenshot_core_loop(True, False)
    threading.Thread(target=run_volume, daemon=True).start()
    return jsonify({"status": "Volume control started"})

@bp.route("/start_screenshot_control", methods=["POST"])
def start_screenshot_control():
    def run_screenshot():
        from services import volume_screenshot_core
        volume_screenshot_core.volume_screenshot_core_loop(False, True)
    threading.Thread(target=run_screenshot, daemon=True).start()
    return jsonify({"status": "Screenshot control started"})

@bp.route("/start_volume_screenshot", methods=["POST"])
def start_volume_screenshot():
    def run_both():
        from services import volume_screenshot_core
        volume_screenshot_core.volume_screenshot_core_loop(True, True)
    threading.Thread(target=run_both, daemon=True).start()
    return jsonify({"status": "Volume + Screenshot control started"})

# --- VOICE / BROWSER CONTROL ---
@bp.route("/start_voice_control", methods=["POST"])
def start_voice_control():
    from services.voice_browser_control import VoiceBrowserController

    print(">>> Initializing controller in main thread")
    vc = VoiceBrowserController()
    print(">>> Running controller in separate thread")
    
    def run_voice():
        print(">>> Voice thread started")
        vc.run()  # make sure microphone init is inside vc.run
        print(">>> Voice thread finished")
    
    threading.Thread(target=run_voice, daemon=False).start()
    return jsonify({"status": "Voice/browser control started"})


