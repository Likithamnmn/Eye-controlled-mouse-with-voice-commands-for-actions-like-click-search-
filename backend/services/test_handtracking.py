import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import HandTrackingModule as htm
    print("HandTrackingModule imported successfully")
    print("Available attributes:", dir(htm))
    
    # Try different possible class names
    possible_names = ['handDetector', 'HandDetector', 'HandTracker', 'handTracker']
    
    for name in possible_names:
        if hasattr(htm, name):
            print(f"✓ Found class: {name}")
            try:
                detector = getattr(htm, name)()
                print(f"✓ Successfully created {name} instance")
            except Exception as e:
                print(f"✗ Error creating {name}: {e}")
        else:
            print(f"✗ No class named: {name}")
            
except Exception as e:
    print(f"Error importing HandTrackingModule: {e}")
    import traceback
    traceback.print_exc()