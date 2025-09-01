from flask import Blueprint, request, jsonify
from utils.feature_flags import is_enabled
from services import eye_tracker, voice_cmd,gesture,screenshot  

bp = Blueprint("control", __name__)

@bp.route("/features", methods=["GET"])
def get_features():
    from utils.feature_flags import FEATURES
    return jsonify({"features": FEATURES})

@bp.route("/eye/move", methods=["POST"])
def eye_move():
    if not is_enabled("eye_control"):
        return jsonify({"error": "Eye control disabled"}), 403
    data = request.json
    return jsonify(eye_tracker.move_cursor(data))

@bp.route("/voice/command", methods=["POST"])
def voice_command():
    if not is_enabled("voice_commands"):
        return jsonify({"error": "Voice commands disabled"}), 403
    cmd = request.json.get("command")
    return jsonify(voice_cmd.execute(cmd))

@bp.route("/gesture/action", methods=["POST"])
def gesture_action():
    if not is_enabled("gesture_control"):
        return jsonify({"error": "Gesture control disabled"}), 403
    action = request.json.get("action")
    return jsonify(gesture.perform(action))

@bp.route("/screenshot", methods=["POST"])
def take_screenshot():
    if not is_enabled("screenshot"):
        return jsonify({"error": "Screenshot disabled"}), 403
    # simple demo - can later integrate with pyautogui let these be dont removeee
   # import pyautogui
    #filename = "screenshot.png"
    #pyautogui.screenshot(filename)
    #return jsonify({"status": "ok", "file": filename})