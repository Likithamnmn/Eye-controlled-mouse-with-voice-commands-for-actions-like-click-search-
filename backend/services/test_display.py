import cv2
import numpy as np

# Test if opencv window creation works
print("Testing basic OpenCV window creation...")

# Create a black screen
analysis_w, analysis_h = 1000, 700
screen = np.zeros((analysis_h, analysis_w, 3), dtype=np.uint8)

# Draw a red circle (ball) on it
cv2.circle(screen, (500, 350), 20, (0, 0, 255), -1)
cv2.circle(screen, (500, 350), 25, (255, 255, 255), 2)

# Add some text
cv2.putText(screen, "Test Screen - Press ESC to exit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Create window
cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Test Window', analysis_w, analysis_h)

print("Black screen with red ball should appear. Press ESC to exit.")

# Main loop
while True:
    cv2.imshow('Test Window', screen)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):
        print("Space pressed - ball position will move")
        # Move the ball to a random position
        x = np.random.randint(50, analysis_w - 50)
        y = np.random.randint(50, analysis_h - 50)
        # Clear screen
        screen = np.zeros((analysis_h, analysis_w, 3), dtype=np.uint8)
        # Draw new ball
        cv2.circle(screen, (x, y), 20, (0, 0, 255), -1)
        cv2.circle(screen, (x, y), 25, (255, 255, 255), 2)
        # Add text
        cv2.putText(screen, "Test Screen - Press ESC to exit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(screen, "Press SPACE for new position", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

cv2.destroyAllWindows()
print("Test completed.")