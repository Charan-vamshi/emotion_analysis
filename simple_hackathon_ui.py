# simple_hackathon_ui.py
import cv2
from deepface import DeepFace
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from collections import defaultdict
from datetime import datetime

class SimpleEmotionDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Behavior AI")
        self.root.geometry("1000x700")
        self.root.configure(bg='#0f1a2b')
        
        # Alert system
        self.alerts = []
        self.alert_thresholds = {
            'stress': 40,      # Angry + Fear + Sad
            'focus': 30,       # Neutral for too long
            'engagement': 70,  # Happy + Surprise
            'extreme_joy': 85  # Very high happiness
        }
        
        # Detection variables
        self.is_running = False
        self.cap = None
        self.current_emotions = []
        self.face_count = 0
        self.engagement_score = 0
        
        self.setup_simple_ui()
        
    def setup_simple_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#0f1a2b', height=80)
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🧠 BEHAVIOR AI", 
                              font=("Arial", 24, "bold"), 
                              fg="white", bg='#0f1a2b')
        title_label.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header_frame, text="● Ready", 
                                    font=("Arial", 12), 
                                    fg="#00ff88", bg='#0f1a2b')
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Control buttons
        control_frame = tk.Frame(header_frame, bg='#0f1a2b')
        control_frame.pack(side=tk.RIGHT)
        
        self.start_btn = tk.Button(control_frame, text="⏺️ Start Recording", 
                                  command=self.start_detection,
                                  font=("Arial", 10, "bold"),
                                  bg="#00ff88", fg="black",
                                  relief="flat", padx=15, pady=5)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#0f1a2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Camera feed
        cam_frame = tk.LabelFrame(main_frame, text="🎥 Live Camera Feed", 
                                 font=("Arial", 12, "bold"),
                                 fg="white", bg='#0f1a2b',
                                 bd=2, relief="groove")
        cam_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.cam_label = tk.Label(cam_frame, 
                                 text="Click Start Recording to begin analysis...", 
                                 bg="black", fg="white", font=("Arial", 12))
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Analytics frame
        analytics_frame = tk.Frame(main_frame, bg='#0f1a2b')
        analytics_frame.pack(fill=tk.X)
        
        # Emotion cards
        cards_frame = tk.Frame(analytics_frame, bg='#0f1a2b')
        cards_frame.pack(fill=tk.X, pady=10)
        
        self.emotion_cards = {}
        emotions = [('😊', 'Happy', '#00ff88'), ('😐', 'Neutral', '#ffaa00'), 
                   ('😠', 'Angry', '#ff4444'), ('😲', 'Surprise', '#00aaff')]
        
        for i, (emoji, emotion, color) in enumerate(emotions):
            card = tk.Frame(cards_frame, bg='#1e2a3d', relief='raised', bd=1)
            card.pack(side=tk.LEFT, expand=True, padx=10, ipadx=15, ipady=10)
            
            tk.Label(card, text=emoji, font=("Arial", 20), 
                    bg='#1e2a3d', fg="white").pack()
            
            value_label = tk.Label(card, text="0%", font=("Arial", 16, "bold"), 
                                  bg='#1e2a3d', fg=color)
            value_label.pack()
            
            tk.Label(card, text=emotion, font=("Arial", 10), 
                    bg='#1e2a3d', fg="white").pack()
            
            self.emotion_cards[emotion.lower()] = value_label
        
        # Stats frame
        stats_frame = tk.Frame(analytics_frame, bg='#0f1a2b')
        stats_frame.pack(fill=tk.X, pady=15)
        
        # Engagement
        engagement_frame = tk.Frame(stats_frame, bg='#0f1a2b')
        engagement_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(engagement_frame, text="ENGAGEMENT:", 
                font=("Arial", 12, "bold"), fg="white", bg='#0f1a2b').pack(anchor="w")
        
        self.engagement_label = tk.Label(engagement_frame, text="0%", 
                                       font=("Arial", 24, "bold"), 
                                       fg="#00ff88", bg='#0f1a2b')
        self.engagement_label.pack(anchor="w")
        
        # Face count
        faces_frame = tk.Frame(stats_frame, bg='#0f1a2b')
        faces_frame.pack(side=tk.LEFT, padx=40)
        
        tk.Label(faces_frame, text="FACES DETECTED:", 
                font=("Arial", 12, "bold"), fg="white", bg='#0f1a2b').pack(anchor="w")
        
        self.faces_label = tk.Label(faces_frame, text="0", 
                                  font=("Arial", 24, "bold"), 
                                  fg="#00ff88", bg='#0f1a2b')
        self.faces_label.pack(anchor="w")
        
        # Alerts frame
        self.alerts_frame = tk.LabelFrame(main_frame, text="🔔 Smart Alerts", 
                                         font=("Arial", 12, "bold"),
                                         fg="white", bg='#0f1a2b')
        self.alerts_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.alerts_container = tk.Frame(self.alerts_frame, bg='#0f1a2b', height=80)
        self.alerts_container.pack(fill=tk.X, padx=10, pady=10)
        self.alerts_container.pack_propagate(False)
        
    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(text="⏹️ Stop Recording", bg="#ff4444")
            self.status_label.config(text="● Recording", fg="#00ff88")
            
            self.camera_thread = threading.Thread(target=self.run_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        else:
            self.stop_detection()
    
    def stop_detection(self):
        self.is_running = False
        self.start_btn.config(text="⏺️ Start Recording", bg="#00ff88")
        self.status_label.config(text="● Ready", fg="#00ff88")
        if self.cap:
            self.cap.release()
    
    def run_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_analysis = 0
        analysis_interval = 2
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            if current_time - last_analysis > analysis_interval:
                try:
                    analysis = DeepFace.analyze(
                        frame, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    self.current_emotions = analysis
                    self.update_analytics(analysis)
                    self.check_alerts(analysis)
                    last_analysis = current_time
                except:
                    self.current_emotions = []
            
            display_frame = self.draw_detections(frame.copy())
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.cam_label.config(image=img_tk)
            self.cam_label.image = img_tk
            
        if self.cap:
            self.cap.release()
    
    def draw_detections(self, frame):
        self.face_count = 0
        total_engagement = 0
        
        for face in self.current_emotions:
            try:
                x = face['region']['x']
                y = face['region']['y']
                w = face['region']['w']
                h = face['region']['h']
                
                emotions = face['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                emotion_score = emotions[dominant_emotion]
                
                # Draw green bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Draw emotion label
                label = f"{dominant_emotion.upper()} ({emotion_score:.0f}%)"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                self.face_count += 1
                
                # Engagement calculation
                engagement_weights = {'happy': 1.0, 'surprise': 0.7, 'neutral': 0.3, 
                                    'sad': 0.1, 'angry': 0.1, 'fear': 0.1, 'disgust': 0.1}
                engagement = emotions.get(dominant_emotion, 0) * engagement_weights.get(dominant_emotion, 0.3)
                total_engagement += engagement
                
            except:
                continue
        
        self.faces_label.config(text=str(self.face_count))
        
        if self.face_count > 0:
            self.engagement_score = total_engagement / self.face_count
        else:
            self.engagement_score = 0
            
        self.engagement_label.config(text=f"{self.engagement_score:.0f}%")
        
        return frame
    
    def update_analytics(self, analysis):
        emotion_totals = defaultdict(float)
        total_faces = len(analysis)
        
        for face in analysis:
            emotions = face['emotion']
            for emotion, score in emotions.items():
                emotion_totals[emotion] += score
        
        for emotion, card in self.emotion_cards.items():
            if total_faces > 0:
                avg_score = emotion_totals.get(emotion, 0) / total_faces
                card.config(text=f"{avg_score:.0f}%")
            else:
                card.config(text="0%")
    
    def check_alerts(self, analysis):
        if not analysis:
            return
            
        emotions = analysis[0]['emotion']
        
        # Stress detection
        stress_level = emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('sad', 0)
        if stress_level > self.alert_thresholds['stress']:
            self.add_alert("🚨 High Stress Detected", f"Stress level: {stress_level:.0f}%")
        
        # Extreme joy
        if emotions.get('happy', 0) > self.alert_thresholds['extreme_joy']:
            self.add_alert("🎉 Extreme Joy Detected", f"Happiness: {emotions['happy']:.0f}%")
        
        # Engagement
        engagement = emotions.get('happy', 0) + emotions.get('surprise', 0)
        if engagement > self.alert_thresholds['engagement']:
            self.add_alert("⭐ High Engagement", f"Engagement: {engagement:.0f}%")
    
    def add_alert(self, title, message):
        alert_frame = tk.Frame(self.alerts_container, bg='#1e2a3d', relief='raised', bd=1)
        alert_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(alert_frame, text=title, 
                font=("Arial", 10, "bold"), 
                fg="#ff4444", bg='#1e2a3d').pack(anchor="w", padx=10, pady=(5, 0))
        
        tk.Label(alert_frame, text=message, 
                font=("Arial", 9), 
                fg="white", bg='#1e2a3d').pack(anchor="w", padx=10, pady=(0, 5))
        
        self.root.after(8000, alert_frame.destroy)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleEmotionDetectionUI(root)
    
    def on_closing():
        app.is_running = False
        if app.cap:
            app.cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()