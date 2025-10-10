import sys, os

# Silence TensorFlow and absl warnings
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass

from flask import Flask
from flask_socketio import SocketIO
from routes.control import bp as control_bp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Register your routes
app.register_blueprint(control_bp, url_prefix="/control")

@app.route("/")
def home():
    return "Server running!"

if __name__ == "__main__":
    print("🚀 Starting Flask-SocketIO server on http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
