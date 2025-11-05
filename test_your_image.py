import cv2
from retinaface import RetinaFace
import os

# Replace with the path to your own image
your_image_path = "C:/Users/Vivek S/Downloads/retinaface-master/your_photo.jpg"  # Using raw string

if os.path.exists(your_image_path):
    print(f"Testing RetinaFace with your image: {your_image_path}")
    try:
        # Try to detect faces in your image
        faces = RetinaFace.detect_faces(your_image_path)
        print(f"Number of faces detected: {len(faces)}")
        
        if faces:
            for face_id, face_data in faces.items():
                print(f"{face_id}: Score={face_data['score']:.4f}, Area={face_data['facial_area']}")
            print("RetinaFace successfully detected faces!")
        else:
            print("No faces detected in your image.")
            
    except Exception as e:
        print(f"Error occurred: {e}")
else:
    print(f"Image not found at: {your_image_path}")
    print("Please place a photo with faces in the current directory and name it 'your_photo.jpg'")
