import cv2
from deepface import DeepFace
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import json

class AnimatedEmotionDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Behavior AI")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0f1a2b')
        
        # Animation states
        self.animation_running = False
        self.pulse_phase = 0
        self.glow_intensity = 0
        self.floating_emojis = []
        
        # Alert system
        self.alerts = []
        self.alert_thresholds = {
            'stress': 40,      # Angry + Fear + Sad
            'focus': 30,       # Neutral for too long
            'engagement': 70,  # Happy + Surprise
            'extreme_joy': 85  # Very high happiness
        }
        
        # Emotion timeline
        self.emotion_history = []
        self.timeline_data = defaultdict(list)
        
        # Detection variables
        self.is_running = False
        self.cap = None
        self.current_emotions = []
        self.face_count = 0
        self.engagement_score = 0
        self.emotion_stats = defaultdict(int)
        self.analysis_count = 0
        
        self.setup_animated_ui()
        self.start_animations()
        
    def setup_animated_ui(self):
        # Header with animated gradient
        header_frame = tk.Frame(self.root, bg='#0f1a2b', height=80)
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        header_frame.pack_propagate(False)
        
        # Animated title
        self.title_label = tk.Label(header_frame, text="🧠 BEHAVIOR AI", 
                                   font=("Arial", 24, "bold"), 
                                   fg="white", bg='#0f1a2b')
        self.title_label.pack(side=tk.LEFT)
        
        # Pulsing status indicator
        self.status_label = tk.Label(header_frame, text="● Ready", 
                                    font=("Arial", 12), 
                                    fg="#00ff88", bg='#0f1a2b')
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Animated control buttons
        control_frame = tk.Frame(header_frame, bg='#0f1a2b')
        control_frame.pack(side=tk.RIGHT)
        
        self.start_btn = AnimatedButton(control_frame, text="⏺️ Start Recording", 
                                       command=self.start_detection,
                                       hover_color="#00cc66",
                                       pulse_when_active=True)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content with smooth layout
        main_frame = tk.Frame(self.root, bg='#0f1a2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Camera and Alerts
        left_panel = tk.Frame(main_frame, bg='#0f1a2b')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera feed with glowing border effect
        self.cam_frame = tk.LabelFrame(left_panel, text="🎥 Live Camera Feed", 
                                      font=("Arial", 12, "bold"),
                                      fg="white", bg='#0f1a2b',
                                      bd=2, relief="groove")
        self.cam_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.cam_label = tk.Label(self.cam_frame, 
                                 text="Click Start Recording to begin analysis...", 
                                 bg="black", fg="white", font=("Arial", 12))
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Smart Alerts Panel with slide-in animation
        self.alerts_frame = tk.LabelFrame(left_panel, text="🔔 Smart Alerts", 
                                         font=("Arial", 12, "bold"),
                                         fg="white", bg='#0f1a2b')
        self.alerts_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.alerts_container = tk.Frame(self.alerts_frame, bg='#0f1a2b', height=100)
        self.alerts_container.pack(fill=tk.X, padx=10, pady=10)
        self.alerts_container.pack_propagate(False)
        
        # Right panel - Analytics and Timeline
        right_panel = tk.Frame(main_frame, bg='#0f1a2b', width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_panel.pack_propagate(False)
        
        # Emotion Timeline Graph
        timeline_frame = tk.LabelFrame(right_panel, text="📈 Emotion Timeline", 
                                      font=("Arial", 12, "bold"),
                                      fg="white", bg='#0f1a2b')
        timeline_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(5, 3), facecolor='#0f1a2b')
        self.ax.set_facecolor('#0f1a2b')
        self.fig.patch.set_alpha(0.0)
        
        # Style the plot
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_ylabel('Intensity %', color='white')
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylim(0, 100)
        
        self.canvas = FigureCanvasTkAgg(self.fig, timeline_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Real-time Analytics with animated cards
        analytics_frame = tk.LabelFrame(right_panel, text="📊 Live Analytics", 
                                       font=("Arial", 12, "bold"),
                                       fg="white", bg='#0f1a2b')
        analytics_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.setup_analytics_cards(analytics_frame)
        
        # Engagement Score with progress animation
        engagement_frame = tk.LabelFrame(right_panel, text="🎯 Engagement Score", 
                                        font=("Arial", 12, "bold"),
                                        fg="white", bg='#0f1a2b')
        engagement_frame.pack(fill=tk.X)
        
        self.engagement_progress = ttk.Progressbar(engagement_frame, 
                                                  orient='horizontal', 
                                                  length=300, 
                                                  mode='determinate')
        self.engagement_progress.pack(padx=10, pady=10, fill=tk.X)
        
        self.engagement_label = tk.Label(engagement_frame, 
                                       text="0%", 
                                       font=("Arial", 16, "bold"), 
                                       fg="#00ff88", bg='#0f1a2b')
        self.engagement_label.pack(pady=(0, 10))
        
    def setup_analytics_cards(self, parent):
        cards_frame = tk.Frame(parent, bg='#0f1a2b')
        cards_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.analytics_cards = {}
        emotions = [
            ('😊', 'Happy', '#00ff88'),
            ('😐', 'Neutral', '#ffaa00'), 
            ('😠', 'Angry', '#ff4444'),
            ('😲', 'Surprise', '#00aaff')
        ]
        
        for i, (emoji, emotion, color) in enumerate(emotions):
            card = AnalyticsCard(cards_frame, emoji, emotion, color)
            card.frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
            self.analytics_cards[emotion.lower()] = card
            
    def start_animations(self):
        self.animation_running = True
        self.animate_elements()
        
    def animate_elements(self):
        if not self.animation_running:
            return
            
        # Pulsing effect for status
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * 3.14159)
        pulse_alpha = 0.5 + 0.5 * abs(np.sin(self.pulse_phase))
        pulse_color = self.fade_color("#00ff88", "#66ffaa", pulse_alpha)
        
        if self.is_running:
            self.status_label.config(fg=pulse_color)
        
        # Glowing border for camera frame
        self.glow_intensity = (self.glow_intensity + 0.05) % 1.0
        glow_alpha = 0.3 + 0.7 * abs(np.sin(self.glow_intensity * 3.14159))
        glow_color = self.fade_color("#00ff88", "#ffffff", glow_alpha)
        
        if self.is_running:
            self.cam_frame.config(fg=glow_color)
        
        # Animate floating emojis
        self.animate_floating_emojis()
        
        # Continue animation loop
        self.root.after(50, self.animate_elements)
        
    def fade_color(self, color1, color2, alpha):
        # Simple color fading between two hex colors
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        r = int(r1 + (r2 - r1) * alpha)
        g = int(g1 + (g2 - g1) * alpha)
        b = int(b1 + (b2 - b1) * alpha)
        
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def animate_floating_emojis(self):
        # Remove old emojis
        for widget in self.alerts_container.winfo_children():
            if hasattr(widget, 'emoji_age'):
                widget.emoji_age += 1
                if widget.emoji_age > 50:  # Remove after 5 seconds
                    widget.destroy()
                    
    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(text="⏹️ Stop Recording", bg="#ff4444")
            self.status_label.config(text="● Recording", fg="#00ff88")
            
            # Start camera thread
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
                    self.update_timeline(analysis)
                    last_analysis = current_time
                except Exception as e:
                    self.current_emotions = []
            
            display_frame = self.draw_animated_detections(frame.copy())
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.cam_label.config(image=img_tk)
            self.cam_label.image = img_tk
            
        if self.cap:
            self.cap.release()
    
    def draw_animated_detections(self, frame):
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
                
                # Animated glowing bounding box
                glow_thickness = 3 + int(2 * abs(np.sin(time.time() * 3)))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), glow_thickness)
                
                # Pulsing emotion label
                label = f"{dominant_emotion.upper()} ({emotion_score:.0f}%)"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                self.face_count += 1
                
                # Engagement calculation
                engagement_weights = {'happy': 1.0, 'surprise': 0.7, 'neutral': 0.3, 
                                    'sad': 0.1, 'angry': 0.1, 'fear': 0.1, 'disgust': 0.1}
                engagement = emotions.get(dominant_emotion, 0) * engagement_weights.get(dominant_emotion, 0.3)
                total_engagement += engagement
                
            except Exception as e:
                continue
        
        self.faces_label.config(text=str(self.face_count))
        
        if self.face_count > 0:
            self.engagement_score = total_engagement / self.face_count
        else:
            self.engagement_score = 0
            
        # Animate engagement progress
        self.animate_engagement_score()
        
        return frame
    
    def animate_engagement_score(self):
        current_progress = self.engagement_progress['value']
        target_progress = self.engagement_score
        
        # Smooth animation towards target
        if abs(current_progress - target_progress) > 1:
            new_progress = current_progress + (target_progress - current_progress) * 0.3
            self.engagement_progress['value'] = new_progress
            self.engagement_label.config(text=f"{new_progress:.0f}%")
            self.root.after(50, self.animate_engagement_score)
        else:
            self.engagement_progress['value'] = target_progress
            self.engagement_label.config(text=f"{target_progress:.0f}%")
    
    def update_analytics(self, analysis):
        emotion_totals = defaultdict(float)
        total_faces = len(analysis)
        
        for face in analysis:
            emotions = face['emotion']
            for emotion, score in emotions.items():
                emotion_totals[emotion] += score
        
        for emotion, card in self.analytics_cards.items():
            if total_faces > 0:
                avg_score = emotion_totals.get(emotion, 0) / total_faces
                card.animate_to_value(avg_score)
            else:
                card.animate_to_value(0)
    
    def check_alerts(self, analysis):
        if not analysis:
            return
            
        emotions = analysis[0]['emotion']
        
        # Stress detection (angry + fear + sad)
        stress_level = emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('sad', 0)
        if stress_level > self.alert_thresholds['stress']:
            self.add_alert("🚨 High Stress Detected", f"Stress level: {stress_level:.0f}%")
        
        # Extreme joy detection
        if emotions.get('happy', 0) > self.alert_thresholds['extreme_joy']:
            self.add_alert("🎉 Extreme Joy Detected", f"Happiness: {emotions['happy']:.0f}%")
        
        # Engagement alert
        engagement = emotions.get('happy', 0) + emotions.get('surprise', 0)
        if engagement > self.alert_thresholds['engagement']:
            self.add_alert("⭐ High Engagement", f"Engagement: {engagement:.0f}%")
    
    def add_alert(self, title, message):
        # Create animated alert
        alert_frame = tk.Frame(self.alerts_container, bg='#1e2a3d', relief='raised', bd=1)
        alert_frame.pack(fill=tk.X, pady=2)
        
        title_label = tk.Label(alert_frame, text=title, 
                              font=("Arial", 10, "bold"), 
                              fg="#ff4444", bg='#1e2a3d')
        title_label.pack(anchor="w", padx=10, pady=(5, 0))
        
        message_label = tk.Label(alert_frame, text=message, 
                               font=("Arial", 9), 
                               fg="white", bg='#1e2a3d')
        message_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # Add animation properties
        alert_frame.emoji_age = 0
        
        # Auto-remove after 8 seconds
        self.root.after(8000, alert_frame.destroy)
    
    def update_timeline(self, analysis):
        if not analysis:
            return
            
        current_time = datetime.now()
        emotions = analysis[0]['emotion']
        
        # Keep only last 20 data points
        if len(self.emotion_history) > 20:
            self.emotion_history.pop(0)
            
        self.emotion_history.append({
            'time': current_time,
            'emotions': emotions
        })
        
        # Update plot
        self.ax.clear()
        self.ax.set_facecolor('#0f1a2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_ylabel('Intensity %', color='white')
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylim(0, 100)
        
        times = [entry['time'].strftime('%H:%M:%S') for entry in self.emotion_history]
        
        colors = {'happy': '#00ff88', 'neutral': '#ffaa00', 
                 'angry': '#ff4444', 'surprise': '#00aaff'}
        
        for emotion, color in colors.items():
            values = [entry['emotions'].get(emotion, 0) for entry in self.emotion_history]
            if len(times) > 1:
                self.ax.plot(times, values, label=emotion.capitalize(), 
                           color=color, linewidth=2, marker='o', markersize=3)
        
        self.ax.legend(facecolor='#1e2a3d', edgecolor='white', labelcolor='white')
        self.canvas.draw()

class AnimatedButton(tk.Button):
    def __init__(self, *args, **kwargs):
        self.hover_color = kwargs.pop('hover_color', '#00cc66')
        self.pulse_when_active = kwargs.pop('pulse_when_active', False)
        self.original_bg = kwargs.get('bg', '#00ff88')
        
        super().__init__(*args, **kwargs)
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        self.config(bg=self.hover_color)
        
    def on_leave(self, e):
        self.config(bg=self.original_bg)

class AnalyticsCard:
    def __init__(self, parent, emoji, emotion, color):
        self.emotion = emotion
        self.color = color
        self.current_value = 0
        self.target_value = 0
        
        self.frame = tk.Frame(parent, bg='#1e2a3d', relief='raised', bd=1)
        
        self.emoji_label = tk.Label(self.frame, text=emoji, 
                                   font=("Arial", 16), 
                                   bg='#1e2a3d', fg="white")
        self.emoji_label.pack(pady=(5, 0))
        
        self.value_label = tk.Label(self.frame, text="0%", 
                                   font=("Arial", 14, "bold"), 
                                   bg='#1e2a3d', fg=color)
        self.value_label.pack()
        
        self.name_label = tk.Label(self.frame, text=emotion, 
                                  font=("Arial", 9), 
                                  bg='#1e2a3d', fg="white")
        self.name_label.pack(pady=(0, 5))
        
    def animate_to_value(self, target_value):
        self.target_value = target_value
        self.animate_step()
        
    def animate_step(self):
        if abs(self.current_value - self.target_value) > 0.5:
            self.current_value += (self.target_value - self.current_value) * 0.3
            self.value_label.config(text=f"{self.current_value:.0f}%")
            self.value_label.after(50, self.animate_step)
        else:
            self.current_value = self.target_value
            self.value_label.config(text=f"{self.target_value:.0f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimatedEmotionDetectionUI(root)
    
    def on_closing():
        app.animation_running = False
        app.is_running = False
        if app.cap:
            app.cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()