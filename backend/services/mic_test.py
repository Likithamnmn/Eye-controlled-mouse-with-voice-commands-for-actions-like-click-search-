"""
Microphone test for speech recognition debugging
"""

import speech_recognition as sr
import time

def test_microphone():
    print("🎤 Quick Microphone Test")
    print("=" * 40)
    
    r = sr.Recognizer()
    
    # Adjust recognition settings for better performance
    r.energy_threshold = 200  # Lower threshold
    r.dynamic_energy_threshold = True
    r.pause_threshold = 0.5   # Shorter pause
    
    print("🔧 Adjusting for ambient noise...")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        print(f"🎚️ Energy threshold set to: {r.energy_threshold}")
    
    print("\n🗣️ Say anything clearly (you have 10 seconds)...")
    print("💡 Try saying: 'hello', 'test', or 'open chrome'")
    
    try:
        with sr.Microphone() as source:
            print("👂 Listening...")
            audio = r.listen(source, timeout=10, phrase_time_limit=5)
        
        print("🔄 Processing speech...")
        text = r.recognize_google(audio)
        print(f"✅ SUCCESS! Recognized: '{text}'")
        print("🎉 Your microphone and speech recognition are working!")
        return True
        
    except sr.WaitTimeoutError:
        print("⏰ No speech detected in 10 seconds")
        print("💡 Try speaking louder or closer to the microphone")
    except sr.UnknownValueError:
        print("❓ Speech detected but couldn't understand it")
        print("💡 Try speaking more clearly or in a quieter environment")
    except sr.RequestError as e:
        print(f"❌ Speech recognition service error: {e}")
        print("💡 Check your internet connection")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    test_microphone()