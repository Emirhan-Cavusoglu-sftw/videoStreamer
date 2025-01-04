# Real-Time Video Streaming with YOLO Person Detection

A comprehensive real-time video streaming application that utilizes GStreamer for video capture and streaming, with integrated YOLO person detection capabilities. The system provides a REST API for external control and includes video compression for enhanced stream quality.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [YOLO Integration](#yolo-integration)
- [Usage Guide](#usage-guide)

## Features
- Real-time video capture from webcam
- YOLO-based person detection with confidence scoring
- REST API for stream control and status monitoring
- Video compression using x264 encoder
- Direct visualization of detection results
- GStreamer pipeline for efficient video streaming
- Configurable resolution and frame rate

## System Architecture

### Key Components
1. **Video Source**
   - Webcam capture using OpenCV
   - Configurable resolution and frame rate
   - DirectShow integration for Windows

2. **Streaming Server**
   - GStreamer pipeline management
   - Video compression
   - Frame processing
   - YOLO detection integration

3. **REST API**
   - Stream control endpoints
   - Status monitoring
   - Detection toggle functionality

## Prerequisites

### 1. System Requirements
- Windows 10/11 (64-bit)
- Python 3.8 or higher
- Minimum 4GB RAM
- Webcam
- Internet connection for initial setup

### 2. Software Dependencies
1. **Python Installation**
   ```bash
   # Download Python 3.x from https://www.python.org/downloads/
   # During installation:
   # ✓ Add Python to PATH
   # ✓ Install pip
   ```

2. **GStreamer Installation**
   ```bash
   # Download from https://gstreamer.freedesktop.org/download/
   # Choose 'Complete' installation
   # Required version: 1.0 or higher
   ```

3. **Environment Variables**
   ```bash
   # Add to System PATH:
   D:\gstreamer\1.0\msvc_x86_64\bin
   ```
This part is for Windows users. There are some errors because of Windows can not find path. So you need add directly.

### 3. YOLO Requirements
Download the following files:

```bash
# YOLOv3 Weights (236MB)
wget https://pjreddie.com/media/files/yolov3.weights

# OLOv3 Configuration
wget https://raw.githubusercontent.com/pjreddie/darknet master/cfg/yolov3.cfg

# COCO Class Names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/video-streaming.git
cd video-streaming
```

### 2. Install Python Dependencies

```bash
pip install opencv-python numpy flask
```

### 3. Project Setup

```bash
# Create project structure
mkdir -p video-streaming
cd video-streaming

# Copy YOLO files
cp path/to/downloads/yolov3.weights .
cp path/to/downloads/yolov3.cfg .
cp path/to/downloads/coco.names .
```

## Project Structure

```bash
video-streaming/
│
├── server.py # Main server implementation
├── client.py # Client implementation
│
├── yolo/ # YOLO related files
│ ├── yolov3.weights # Model weights
│ ├── yolov3.cfg # Model configuration
│ └── coco.names # Class names
│
└── README.md # Documentation
```

## Implementation Details

### Server Implementation (server.py)

#### 1. VideoStream Class

```python
class VideoStream:
def init(self):
self.resolution = (640, 480)
self.framerate = 30
self.running = False
self.enable_detection = False
```

Key methods:
- `start()`: Initializes video capture and streaming
- `stop()`: Gracefully stops the stream
- `detect_persons()`: Performs YOLO detection
- `create_gstreamer_pipeline()`: Sets up streaming pipeline

#### 2. REST API Implementation

```python
from flask import Flask, jsonify, request
app = Flask(name)
@app.route('/start', methods=['POST'])
def start_stream():
```

### YOLO Integration

#### 1. Model Loading

```python
def load_yolo(self):
self.net = cv2.dnn.readNet(
"yolov3.weights",
"yolov3.cfg"
)
```

#### 2. Detection Process

```python
def detect_persons(self, frame):
# Create blob from frame
blob = cv2.dnn.blobFromImage(
frame,
1/255.0,
(416, 416),
swapRB=True
)
```

## Usage Guide

### 1. Starting the Server

```bash
cd video-streaming
python server.py
```

### 2. API Commands

```bash
# Start streaming
curl -X POST http://localhost:8000/start
# Enable YOLO detection
curl -X POST -H "Content-Type: application/json" \
-d '{"enable":true}' \
http://localhost:8000/detection
```

### 3. Starting the Client

```bash
cd video-streaming
python client.py
```

### 4. Check status

```bash
curl http://localhost:8000/status
```

### 5. Stop streaming

```bash
curl -X POST http://localhost:8000/stop
```