import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 Camera started - Press 'q' to quit")

last_analysis_time = 0
analysis_interval = 2  # Analyze every 2 seconds
current_emotions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    # Analyze emotions every 2 seconds
    if current_time - last_analysis_time > analysis_interval:
        try:
            analysis = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='opencv'
            )
            current_emotions = analysis
            last_analysis_time = current_time
        except:
            current_emotions = []
    
    # Display emotions from the last analysis
    for face in current_emotions:
        try:
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']
            
            emotions = face['emotion']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_score = emotions[dominant_emotion]
            
            # Draw green rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display emotion text
            text = f"{dominant_emotion} ({emotion_score:.1f}%)"
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except:
            pass
    
    cv2.imshow('Emotion Detection - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program ended")