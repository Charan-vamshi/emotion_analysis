import cv2
import numpy as np
from retinaface import RetinaFace
import urllib.request
import os

# Download a sample image with faces
image_url = "https://raw.githubusercontent.com/serengil/retinaface/master/tests/dataset/img1.jpg"
image_path = "real_test_image.jpg"

if not os.path.exists(image_path):
    print("Downloading test image...")
    urllib.request.urlretrieve(image_url, image_path)
    print("Download completed!")

print("Testing RetinaFace with real image...")
try:
    # Try to detect faces in the real image
    faces = RetinaFace.detect_faces(image_path)
    print(f"Number of faces detected: {len(faces)}")
    
    if faces:
        for face_id, face_data in faces.items():
            print(f"{face_id}: Score={face_data['score']:.4f}, Area={face_data['facial_area']}")
        print("RetinaFace is working perfectly!")
    else:
        print("No faces detected in the image.")
        
except Exception as e:
    print(f"Error occurred: {e}")
