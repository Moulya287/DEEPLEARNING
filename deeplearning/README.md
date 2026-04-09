# SMART-HAMS: Smart Highway Anomaly Monitoring System

## Project Title
**SMART-HAMS** - An AI-Powered Smart Highway Anomaly Monitoring System for Real-Time Accident, Wildlife, and Hazard Detection

## Abstract
This project presents an AI-powered framework for real-time detection of multiple hazards on highways including accidents, wildlife crossings, fire outbreaks, traffic jams, and crowd formation. The system integrates YOLOv8 for vehicle and person detection with Faster R-CNN for wildlife detection, processing video streams at 20 frames per second. A time-based risk classification mechanism categorizes detected events as LOW, MEDIUM, or HIGH risk based on stationary duration and object size. The system achieves 82.1% average detection accuracy across all hazard types.

## Features
- Vehicle Detection with Risk Assessment (LOW/MEDIUM/HIGH)
- Wildlife Detection (Elephants, Bears, Cows, Horses, Giraffes)
- Accident Detection (Vehicle stopped >8 seconds triggers alert)
- Fire Detection using HSV color space analysis
- Traffic Jam Detection (when >15 vehicles in frame)
- Crowd Detection (when >10 people in frame)
- Color-coded bounding boxes (Green/Yellow/Red)
- Flask-based Web Interface for easy upload and viewing

## Technologies Used
- Python 3.9+
- OpenCV for video frame extraction and processing
- YOLOv8 for vehicle and person detection
- Faster R-CNN (ResNet-50 FPN) for wildlife detection
- Flask for web interface
- FFmpeg for video format conversion

## Project Structure
DEEPLEARNING/
│
├── app.py # Flask web application
├── detect.py # Detection script (YOLOv8 + Faster R-CNN)
├── yolov8n.pt # YOLOv8 model weights
├── report.pdf # IEEE research paper
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── templates/
│ ├── upload.html # Video upload page
│ └── result.html # Results display page
│
├── uploads/ # Temporary storage for uploaded videos
└── outputs/ # Processed video output storage


## Detection Capabilities

### Vehicle Detection
- Detects: car, truck, bus, motorcycle
- Tracks stationary duration
- Risk levels based on stopped time

### Wildlife Detection
- Detects: elephant, bear, giraffe, horse, cow, zebra, dog, cat, bird
- Size-based risk classification
- Large animals (>15% frame) trigger HIGH alert

### Fire Detection
- HSV color space analysis
- Fire pixel threshold: 8000 pixels
- Displays pixel count in alert

### Traffic Analysis
- Traffic jam: >15 vehicles
- Crowd: >10 persons
- Accident: vehicle stopped >8 seconds

## Risk Classification

| Risk Level | Vehicle Condition | Wildlife Condition | Box Color |
|------------|-------------------|-------------------|-----------|
| LOW | Moving normally | Small animal | Green |
| MEDIUM | Stopped 5-8 seconds | Medium animal (8-15% frame) | Yellow |
| HIGH | Stopped >8 seconds | Large animal (>15% frame) | Red |

## Performance Metrics

| Hazard Type | Test Cases | Accuracy |
|-------------|------------|----------|
| Accident | 5 | 80.0% |
| Wildlife (Large) | 8 | 87.5% |
| Fire | 4 | 75.0% |
| Traffic Jam | 6 | 83.3% |
| Crowd | 5 | 80.0% |
| **Average** | **28** | **82.1%** |

## Output Examples

The system produces annotated videos with:
- Bounding boxes around detected objects (color-coded by risk)
- Alert text for critical events (ACCIDENT, FIRE DETECTED, ELEPHANT ON ROAD)
- Frame counter and vehicle count display
- Fire pixel count when fire detected

## Authors
- **Moulya N** - School of Computer Science, R V University
- **Nama Shritha** - School of Computer Science, R V University
- **Nireeksha P Rathod** - School of Computer Science, R V University
- **Pooja Naresh** - School of Computer Science, R V University

**Project Guide:** Shoeb Ahmad

## Institution
**R V University**
Bangalore, India
Department of Computer Science
B.Tech (Hons) Artificial Intelligence and Machine Learning

## Date
April 2026

## Acknowledgments
The authors thank R V University Department of Computer Science for providing computational resources and laboratory facilities. Special thanks to the project guide Shoeb Ahmad for continuous guidance and support throughout this research.

## License
This project is for academic submission purposes only.