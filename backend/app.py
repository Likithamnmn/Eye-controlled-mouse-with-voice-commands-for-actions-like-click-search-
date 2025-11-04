from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from routes.control import bp as control_bp
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ✅ Suppress TensorFlow & absl warnings safely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass

# ✅ Initialize Flask
app = Flask(__name__)

# ✅ Handle OPTIONS preflight at APP LEVEL (before blueprint registration)
@app.before_request
def handle_preflight():
    """Handle OPTIONS preflight requests for all routes"""
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        response.headers.add("Access-Control-Max-Age", "3600")
        return response

# ✅ Enable CORS with automatic OPTIONS handling
CORS(
    app,
    origins=["http://localhost:3000"],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True,
    automatic_options=True
)

# ✅ Register blueprint
app.register_blueprint(control_bp, url_prefix="/control")

# ✅ Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins=["http://localhost:3000"],
    async_mode="threading"
)

@app.route("/")
def home():
    return "Server running!"

if __name__ == "__main__":
    print("🚀 Starting Flask-SocketIO server on http://127.0.0.1:5000")
    print("\n📋 Registered routes:")
    for rule in app.url_map.iter_rules():
        if 'control' in rule.rule:
            print(f"  {rule.rule} [{', '.join(rule.methods)}]")
    print()
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )