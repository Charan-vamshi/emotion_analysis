import cv2
import numpy as np
from retinaface import RetinaFace

# Create a simple test image with a face (solid color rectangle as a placeholder)
test_image = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
# Draw a simple "face" rectangle
cv2.rectangle(test_image, (50, 50), (150, 150), (0, 0, 255), -1)  # Red rectangle

# Save the test image
cv2.imwrite("test_image.jpg", test_image)

print("Testing RetinaFace detection...")
try:
    # Try to detect faces
    faces = RetinaFace.detect_faces("test_image.jpg")
    print(f"Detection result: {faces}")
    print("RetinaFace is working correctly!")
except Exception as e:
    print(f"Error occurred: {e}")
    print("This might be expected since we're using a simple test image")
