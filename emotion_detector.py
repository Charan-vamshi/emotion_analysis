import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
from deepface import DeepFace
import threading
import time
from PIL import Image, ImageTk
from collections import defaultdict
import pandas as pd
from datetime import datetime

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Behavior Analysis - Facial Expression Recognition")
        self.root.geometry("1400x900")
        
        # ========== ENHANCED CONFIGURATION ==========
        self.confidence_threshold = 80  # Face recognition confidence
        self.cooldown_period = 30  # Seconds between logs
        self.cooldown_dict = defaultdict(float)
        self.emotion_history = []
        # ============================================
        
        # Variables
        self.is_running = False
        self.cap = None
        self.recognition_count = 0
        self.unique_faces = set()
        self.confidence_values = []
        
        # Create UI
        self.create_enhanced_widgets()
        
    def create_enhanced_widgets(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_container, 
                              text="AI BEHAVIOR ANALYSIS - Facial Expression Recognition System", 
                              font=("Arial", 18, "bold"), bg="#1e3d59", fg="white")
        title_label.pack(fill=tk.X, pady=(0, 20))
        
        # Content frame
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Video feed and Emotions
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video feed
        video_frame = ttk.LabelFrame(left_frame, text="Live Camera Feed", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(video_frame, text="Initializing Camera...", 
                                   bg="black", fg="white", font=("Arial", 12))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Real-time Emotions
        emotion_frame = ttk.LabelFrame(left_frame, text="Real-time Emotion Analysis", padding=10)
        emotion_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Emotion bars
        self.emotion_bars = {}
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        for i, emotion in enumerate(emotions):
            emotion_row = ttk.Frame(emotion_frame)
            emotion_row.pack(fill=tk.X, pady=2)
            
            tk.Label(emotion_row, text=emotion.capitalize(), width=10, anchor="w").pack(side=tk.LEFT)
            
            # Progress bar for emotion intensity
            self.emotion_bars[emotion] = ttk.Progressbar(emotion_row, orient=tk.HORIZONTAL, length=200)
            self.emotion_bars[emotion].pack(side=tk.LEFT, padx=5)
            
            # Percentage label
            self.emotion_bars[emotion + '_label'] = tk.Label(emotion_row, text="0%", width=5)
            self.emotion_bars[emotion + '_label'].pack(side=tk.LEFT)
        
        # Right side - Controls and Analytics
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Controls
        controls_frame = ttk.LabelFrame(right_frame, text="System Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(controls_frame, text="Start Analysis", 
                                   command=self.start_recognition, width=20)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(controls_frame, text="Stop Analysis", 
                                  command=self.stop_recognition, state=tk.DISABLED, width=20)
        self.stop_btn.pack(pady=5)
        
        # Export button
        self.export_btn = ttk.Button(controls_frame, text="Export Data", 
                                    command=self.export_data, width=20)
        self.export_btn.pack(pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(right_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="Status: Ready", 
                                    font=("Arial", 11), fg="blue")
        self.status_label.pack()
        
        # Face Recognition Stats
        stats_frame = ttk.LabelFrame(right_frame, text="Face Recognition Analytics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Statistics labels
        stats_data = [
            ("Total Recognitions:", "total_label"),
            ("Unique Faces:", "unique_label"), 
            ("Success Rate:", "rate_label"),
            ("Avg Confidence:", "confidence_label"),
            ("Dominant Emotion:", "dominant_emotion_label")
        ]
        
        for i, (text, var_name) in enumerate(stats_data):
            tk.Label(stats_grid, text=text, font=("Arial", 9)).grid(row=i, column=0, sticky="w", pady=2)
            setattr(self, var_name, tk.Label(stats_grid, text="0", font=("Arial", 9, "bold")))
            getattr(self, var_name).grid(row=i, column=1, sticky="e", padx=(10, 0))
        
        # Emotion History
        emotion_stats_frame = ttk.LabelFrame(right_frame, text="Emotion History", padding=10)
        emotion_stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        emotion_stats_grid = ttk.Frame(emotion_stats_frame)
        emotion_stats_grid.pack(fill=tk.X)
        
        self.emotion_stats = {}
        emotions_display = ['happy', 'neutral', 'sad', 'angry']
        
        for i, emotion in enumerate(emotions_display):
            tk.Label(emotion_stats_grid, text=f"{emotion.capitalize()}:", font=("Arial", 9)).grid(row=i, column=0, sticky="w", pady=1)
            self.emotion_stats[emotion] = tk.Label(emotion_stats_grid, text="0%", font=("Arial", 9, "bold"))
            self.emotion_stats[emotion].grid(row=i, column=1, sticky="e", padx=(10, 0))
        
        # Recognition Log
        log_frame = ttk.LabelFrame(right_frame, text="Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_container, height=12, width=45, font=("Consolas", 8))
        scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize labels
        self.total_label.config(text="0")
        self.unique_label.config(text="0")
        self.rate_label.config(text="0%")
        self.confidence_label.config(text="0%")
        self.dominant_emotion_label.config(text="None")
        
    def start_recognition(self):
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Analyzing Behavior...", fg="green")
        self.log_message("AI Behavior Analysis started")
        
        # Start recognition in separate thread
        self.recognition_thread = threading.Thread(target=self.run_recognition)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
    def stop_recognition(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", fg="red")
        self.log_message("Behavior Analysis stopped")
        
        if self.cap:
            self.cap.release()
        
    def log_message(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        self.log_text.insert(tk.END, f"{timestamp} {message}\n")
        self.log_text.see(tk.END)
        
    def update_emotion_display(self, emotions):
        """Update emotion progress bars and labels"""
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            self.dominant_emotion_label.config(text=dominant_emotion.capitalize())
            
            for emotion, score in emotions.items():
                if emotion in self.emotion_bars:
                    percentage = int(score)
                    self.emotion_bars[emotion]['value'] = percentage
                    self.emotion_bars[emotion + '_label'].config(text=f"{percentage}%")
        
    def run_recognition(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_message("ERROR: Could not access camera")
            return
            
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.last_process = time.time()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Resize frame for display
                frame = cv2.resize(frame, (800, 600))
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)
                self.video_label.config(image=img_tk, text="")
                self.video_label.image = img_tk
                
                # Run recognition every 2 seconds
                if time.time() - self.last_process > 2:
                    self.process_frame(frame)
                
            self.root.update()
        
        if self.cap:
            self.cap.release()
        
    def process_frame(self, frame):
        try:
            recognition_made = False
            current_emotions = None
            
            # ========== FACE RECOGNITION (Original Feature) ==========
            if os.path.exists("face_database") and len(os.listdir("face_database")) > 0:
                results = DeepFace.find(
                    img_path=frame,
                    db_path="face_database",
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )
                
                if results and not results[0].empty:
                    for _, row in results[0].iterrows():
                        name = os.path.basename(row['identity']).split('.')[0]
                        confidence = (1 - row['distance']) * 100
                        
                        # Apply confidence threshold
                        if confidence < self.confidence_threshold:
                            continue
                        
                        current_time = time.time()
                        last_recognition = self.cooldown_dict.get(name, 0)
                        
                        # Apply cooldown period
                        if current_time - last_recognition < self.cooldown_period:
                            continue
                        
                        # Update cooldown time
                        self.cooldown_dict[name] = current_time
                        recognition_made = True
                        
                        # Update statistics
                        self.recognition_count += 1
                        self.unique_faces.add(name)
                        self.confidence_values.append(confidence)
                        
                        # Update UI
                        self.total_label.config(text=str(self.recognition_count))
                        self.unique_label.config(text=str(len(self.unique_faces)))
                        
                        if self.confidence_values:
                            avg_confidence = sum(self.confidence_values) / len(self.confidence_values)
                            self.confidence_label.config(text=f"{avg_confidence:.1f}%")
                        
                        # Log the recognition
                        self.log_message(f"Recognized: {name} ({confidence:.1f}%)")
            
            # ========== EMOTION ANALYSIS (New Feature) ==========
            try:
                emotion_analysis = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if emotion_analysis:
                    emotions = emotion_analysis[0]['emotion']
                    current_emotions = emotions
                    
                    # Update emotion display
                    self.update_emotion_display(emotions)
                    
                    # Store emotion history
                    self.emotion_history.append({
                        'timestamp': datetime.now(),
                        'emotions': emotions,
                        'dominant_emotion': max(emotions.items(), key=lambda x: x[1])[0]
                    })
                    
                    # Update emotion statistics (last 10 readings)
                    recent_emotions = self.emotion_history[-10:]
                    if recent_emotions:
                        for emotion in ['happy', 'neutral', 'sad', 'angry']:
                            avg_score = sum(entry['emotions'].get(emotion, 0) for entry in recent_emotions) / len(recent_emotions)
                            self.emotion_stats[emotion].config(text=f"{avg_score:.1f}%")
                    
                    # Log significant emotion changes
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                    if emotions[dominant_emotion] > 70:  # Only log strong emotions
                        if not recognition_made:  # Don't double-log if person was recognized
                            self.log_message(f"Emotion: {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.1f}%)")
                            
            except Exception as e:
                pass  # Silent fail for emotion analysis
            
            self.last_process = time.time()
            
        except Exception as e:
            self.log_message(f"System Error: {str(e)}")
    
    def export_data(self):
        """Export recognition and emotion data to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"behavior_analysis_{timestamp}.csv"
            
            # Prepare data for export
            data = []
            for entry in self.emotion_history:
                row = {'timestamp': entry['timestamp']}
                row.update(entry['emotions'])
                row['dominant_emotion'] = entry['dominant_emotion']
                data.append(row)
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                self.log_message(f"Data exported to {filename}")
                messagebox.showinfo("Export Successful", f"Data exported to {filename}")
            else:
                messagebox.showwarning("No Data", "No emotion data to export")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()