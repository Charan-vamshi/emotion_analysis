import cv2
import numpy as np
from retinaface import RetinaFace

# Let's try using the OpenCV built-in function to create a simple face-like pattern
# Create a more complex pattern that might resemble a face
test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255  # White background

# Draw a face-like pattern
# Face oval
cv2.ellipse(test_image, (150, 150), (100, 130), 0, 0, 360, (200, 150, 150), -1)

# Eyes
cv2.circle(test_image, (120, 120), 15, (0, 0, 0), -1)  # Left eye
cv2.circle(test_image, (180, 120), 15, (0, 0, 0), -1)  # Right eye

# Nose
cv2.ellipse(test_image, (150, 160), (10, 15), 0, 0, 360, (0, 0, 0), -1)

# Mouth
cv2.ellipse(test_image, (150, 200), (40, 20), 0, 0, 180, (0, 0, 0), 2)

# Save the test image
cv2.imwrite("face_pattern.jpg", test_image)

print("Testing RetinaFace with face-like pattern...")
try:
    # Try to detect faces in our pattern
    faces = RetinaFace.detect_faces("face_pattern.jpg")
    print(f"Number of faces detected: {len(faces)}")
    
    if faces:
        for face_id, face_data in faces.items():
            print(f"{face_id}: Score={face_data['score']:.4f}, Area={face_data['facial_area']}")
        print("RetinaFace detected our pattern as a face!")
    else:
        print("No faces detected in the pattern image.")
        
except Exception as e:
    print(f"Error occurred: {e}")
