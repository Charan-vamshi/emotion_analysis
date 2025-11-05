import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2
import os
from deepface import DeepFace
import time

print("HIGH ACCURACY Face Recognition")
print("Using optimal settings for better matching")

cap = cv2.VideoCapture(0)
last_time = 0
current_results = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()

    if current_time - last_time > 2:  # More time for better processing
        try:
            if os.path.exists("face_database") and len(os.listdir("face_database")) > 0:
                # OPTIMAL SETTINGS FOR ACCURACY
                results = DeepFace.find(
                    img_path=frame,
                    db_path="face_database",
                    model_name="VGG-Face",  # Better for accuracy
                    detector_backend="opencv",  # Fastest detection
                    distance_metric="cosine",  # Better matching
                    enforce_detection=False,
                    silent=True
                )
                
                current_results = {}
                if len(results) > 0 and not results[0].empty:
                    for _, row in results[0].iterrows():
                        x, y, w, h = int(row['source_x']), int(row['source_y']), int(row['source_w']), int(row['source_h'])
                        
                        # Check multiple distance metrics
                        distance = row.get('distance', row.get('cosine', 1.0))
                        
                        if distance < 0.6:  # Better threshold
                            identity = os.path.basename(row['identity']).split('.')[0]
                            confidence = f"{(1 - distance) * 100:.1f}%"
                            
                            if (1 - distance) * 100 > 60:  # Only show if >60% confidence
                                face_key = f"{x}_{y}_{w}_{h}"
                                current_results[face_key] = (x, y, w, h, identity, confidence)
                
                last_time = current_time
                
        except Exception as e:
            print(f"Error: {e}")
            current_results = {}
    
    # Display only high-confidence results
    for (x, y, w, h, name, confidence) in current_results.values():
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('High Accuracy Recognition - Press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()