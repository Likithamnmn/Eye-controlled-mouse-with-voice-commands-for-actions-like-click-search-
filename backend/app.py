from flask import Flask, jsonify
from flask_socketio import SocketIO
from routes.control import bp as control_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = '239541moshimoshi'

# Socket.IO setup
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Register routes
app.register_blueprint(control_bp, url_prefix="/api")

# Default route
@app.route("/")
def home():
    return jsonify({"message": "Backend with Flask + SocketIO is running!"})

# Socket events
@socketio.on("connect")
def handle_connect():
    print("User connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("User disconnected")

@socketio.on("eye_event")
def handle_eye_event(data):
    print("Eye Event:", data)
    socketio.emit("eye_update", {"status": "moved", "coords": data})

@socketio.on("voice_event")
def handle_voice_event(data):
    print("Voice Command:", data)
    socketio.emit("voice_update", {"status": "executed", "command": data})

if __name__ == "__main__":
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
