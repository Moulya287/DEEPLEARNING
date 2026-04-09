import cv2
import time
import sys
import torch
import os
import warnings
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms

# Suppress warnings
warnings.filterwarnings("ignore")

# Get command line arguments
input_path = sys.argv[1]
output_path = sys.argv[2]

# Load models
print("Loading AI models...")
model_main = YOLO("yolov8n.pt")
wild_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
wild_model.eval()

transform = transforms.Compose([transforms.ToTensor()])

# Open video
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (640,480))

# Tracking variables
object_data = {}
previous_positions = {}
frame_count = 0
wild_results = []
last_alert_time = {}  # To avoid repeated alerts
ALERT_COOLDOWN = 30  # Frames between same alert

# ALERT FLAGS
alert_triggered = {
    "accident": False,
    "traffic_jam": False,
    "fire": False,
    "wildlife_high": False,
    "crowd": False
}

# FIXED THRESHOLDS - Less sensitive
MOVEMENT_THRESHOLD = 30  # Increased from 20
STATIONARY_TIME_HIGH = 15  # Increased from 10 seconds
STATIONARY_TIME_MEDIUM = 8  # Increased from 5 seconds
TRAFFIC_JAM_THRESHOLD = 15  # Increased from 10 vehicles
CROWD_THRESHOLD = 10  # Increased from 8
FIRE_PIXEL_THRESHOLD = 8000  # Increased from 5000

# Animal classes (COCO dataset)
COCO_ANIMALS = {
    16: "bird", 17: "cat", 18: "dog", 19: "horse",
    20: "sheep", 21: "cow", 22: "elephant",
    23: "bear", 24: "zebra", 25: "giraffe"
}

# Large animals that need special attention
LARGE_ANIMALS = ["elephant", "bear", "giraffe", "horse", "cow"]

def is_stationary(prev, curr):
    """Check if object is stationary"""
    return abs(prev[0]-curr[0]) < MOVEMENT_THRESHOLD and abs(prev[1]-curr[1]) < MOVEMENT_THRESHOLD

def detect_wildlife(frame):
    """Detect wildlife in frame"""
    img = transform(frame)
    with torch.no_grad():
        preds = wild_model([img])[0]
    
    results = []
    for i in range(len(preds['labels'])):
        if preds['scores'][i] > 0.75:  # Increased threshold for accuracy
            results.append((preds['labels'][i].item(), preds['boxes'][i].tolist()))
    return results

def check_fire(frame):
    """Detect fire in frame - IMPROVED with better filtering"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # More specific fire color range
    lower_fire = np.array([0, 100, 150])
    upper_fire = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    
    # Additional orange/red range for fire
    lower_fire2 = np.array([5, 150, 150])
    upper_fire2 = np.array([15, 255, 255])
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    
    mask = cv2.bitwise_or(mask, mask2)
    
    # Remove small noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    fire_pixels = cv2.countNonZero(mask)
    return fire_pixels > FIRE_PIXEL_THRESHOLD, fire_pixels

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640,480))
    frame_count += 1
    current_time = time.time()
    
    # Run YOLO detection every frame
    results = model_main(frame, verbose=False)
    
    # Run wildlife detection every 15 frames (reduced frequency)
    if frame_count % 15 == 0:
        wild_results = detect_wildlife(frame)
    
    # Counters
    vehicle_count = 0
    person_count = 0
    alert_this_frame = False
    
    # Process YOLO detections
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model_main.names[cls]
            
            # Vehicle and person detection
            if label in ["car", "truck", "bus", "motorcycle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1+x2)//2, (y1+y2)//2)
                
                obj_id = f"{label}_{x1}_{y1}_{x2}_{y2}"
                
                # Initialize tracking
                if obj_id not in object_data:
                    object_data[obj_id] = {"start": current_time, "pos": center, "alerted": False}
                
                prev = object_data[obj_id]["pos"]
                
                # Calculate stationary duration
                if is_stationary(prev, center):
                    duration = current_time - object_data[obj_id]["start"]
                else:
                    object_data[obj_id]["start"] = current_time
                    duration = 0
                    object_data[obj_id]["alerted"] = False
                
                object_data[obj_id]["pos"] = center
                
                # Risk assessment with FIXED thresholds
                if duration > STATIONARY_TIME_HIGH:
                    risk = "HIGH"
                    color = (0, 0, 255)
                    # Trigger accident alert for vehicles stopped too long
                    if not object_data[obj_id].get("alerted", False):
                        object_data[obj_id]["alerted"] = True
                        alert_triggered["accident"] = True
                        alert_this_frame = True
                elif duration > STATIONARY_TIME_MEDIUM:
                    risk = "MEDIUM"
                    color = (0, 255, 255)
                else:
                    risk = "LOW"
                    color = (0, 255, 0)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} [{risk}]", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Show stationary time
                if duration > 2:
                    cv2.putText(frame, f"Stopped: {duration:.0f}s", (x1, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Track position history
                if obj_id in previous_positions:
                    prev_pos = previous_positions[obj_id]
                    speed = np.sqrt((center[0]-prev_pos[0])**2 + (center[1]-prev_pos[1])**2)
                    if speed < 2 and duration > STATIONARY_TIME_HIGH:
                        cv2.putText(frame, "⚠️ ACCIDENT ⚠️", (200, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                previous_positions[obj_id] = center
                vehicle_count += 1
                
            elif label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "person", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Wildlife detection with size-based risk
    for label_id, box in wild_results:
        if label_id in COCO_ANIMALS:
            name = COCO_ANIMALS[label_id]
            x1, y1, x2, y2 = map(int, box)
            
            size = (x2-x1) * (y2-y1)
            frame_area = 640 * 480
            
            # Size-based risk (adjusted thresholds)
            if name in LARGE_ANIMALS:
                if size > frame_area * 0.15:  # Large animal covering 15% of frame
                    risk = "HIGH"
                    color = (0, 0, 255)
                    if not alert_triggered["wildlife_high"]:
                        alert_triggered["wildlife_high"] = True
                        alert_this_frame = True
                elif size > frame_area * 0.08:
                    risk = "MEDIUM"
                    color = (0, 165, 255)
                else:
                    risk = "LOW"
                    color = (255, 0, 255)
            else:
                # Small animals (birds, cats, dogs)
                if size > frame_area * 0.1:
                    risk = "MEDIUM"
                    color = (0, 165, 255)
                else:
                    risk = "LOW"
                    color = (255, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name.upper()} [{risk}]", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Special alert for large animal on road
            if risk == "HIGH":
                cv2.putText(frame, f"🐘 {name.upper()} ON ROAD!", (150, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Traffic Jam Detection (FIXED threshold)
    if vehicle_count > TRAFFIC_JAM_THRESHOLD:
        cv2.putText(frame, f"🚗 TRAFFIC JAM - {vehicle_count} vehicles", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if not alert_triggered["traffic_jam"]:
            alert_triggered["traffic_jam"] = True
            alert_this_frame = True
    else:
        alert_triggered["traffic_jam"] = False
    
    # Crowd Detection (FIXED threshold)
    if person_count > CROWD_THRESHOLD:
        cv2.putText(frame, f"👥 CROWD DETECTED - {person_count} people", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        if not alert_triggered["crowd"]:
            alert_triggered["crowd"] = True
            alert_this_frame = True
    else:
        alert_triggered["crowd"] = False
    
    # Fire Detection (IMPROVED)
    fire_detected, fire_pixels = check_fire(frame)
    if fire_detected:
        cv2.putText(frame, f"🔥 FIRE DETECTED! ({fire_pixels} pixels)", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
        if not alert_triggered["fire"]:
            alert_triggered["fire"] = True
            alert_this_frame = True
    else:
        alert_triggered["fire"] = False
    
    # Cooldown for wildlife alert
    if frame_count % ALERT_COOLDOWN == 0:
        alert_triggered["wildlife_high"] = False
    
    # Display frame info
    cv2.putText(frame, f"Frame: {frame_count} | Vehicles: {vehicle_count} | People: {person_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show active alerts
    y_offset = 450
    if any(alert_triggered.values()):
        cv2.putText(frame, "🚨 ACTIVE ALERTS 🚨", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    out.write(frame)

# Cleanup
cap.release()
out.release()

# Convert for web playback
print("Converting video for playback...")
os.system(f'ffmpeg -y -i "{output_path}" -vcodec libx264 -movflags +faststart "outputs/final_output.mp4" 2>nul')

print("✅ Processing complete! Output saved to outputs/final_output.mp4")