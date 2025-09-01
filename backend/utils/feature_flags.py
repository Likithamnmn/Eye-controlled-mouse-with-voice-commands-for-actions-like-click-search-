# Define available features
FEATURES = {
    "eye_control": True,
    "voice_commands": True,
    "gesture_control": False,
    "screenshot": True,
}

def is_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled."""
    return FEATURES.get(feature_name, False)
