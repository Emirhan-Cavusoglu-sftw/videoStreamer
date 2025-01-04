from flask import Flask, jsonify, request
import subprocess
import logging
import threading
import cv2
import numpy as np
import os
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class VideoStream:
    def __init__(self):
        self.running = False
        self.thread = None
        self.process = None
        self.compression_level = 23
        self.resolution = (640, 480)
        self.framerate = 30
        
        # YOLO modelini yükle
        self.enable_detection = True
        self.load_yolo()
        
        # OpenCV video capture
        self.cap = None

    def load_yolo(self):
        try:
            # Dosya yollarını kontrol et
            weights_path = "yolov3.weights"
            config_path = "yolov3.cfg"
            names_path = "coco.names"
            
            # Dosyaların varlığını kontrol et
            if not os.path.exists(weights_path):
                logger.error(f"YOLO weights file not found: {weights_path}")
                self.enable_detection = False
                return
            
            if not os.path.exists(config_path):
                logger.error(f"YOLO config file not found: {config_path}")
                self.enable_detection = False
                return
            
            if not os.path.exists(names_path):
                logger.error(f"COCO names file not found: {names_path}")
                self.enable_detection = False
                return
            
            logger.info("Loading YOLO model...")
            self.net = cv2.dnn.readNet(weights_path, config_path)
            logger.info("YOLO network loaded")
            
            self.classes = []
            with open(names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.classes)} classes")
            
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            logger.info("YOLO model loaded successfully")
            logger.info("Detection is enabled")
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            logger.error("Error details:", exc_info=True)
            self.enable_detection = False

    def detect_persons(self, frame):
        if not self.enable_detection:
            return frame
            
        try:
            height, width, _ = frame.shape
            
            # Blob parametrelerini ayarla
            blob = cv2.dnn.blobFromImage(
                frame, 
                scalefactor=1/255.0,  # Normalize et
                size=(320, 320),      # Daha küçük boyut, daha hızlı işlem
                mean=(0, 0, 0),       # Mean subtraction
                swapRB=True,          # BGR -> RGB
                crop=False
            )
            
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            boxes = []
            confidences = []
            
            # Güven eşiğini düşür
            confidence_threshold = 0.4  # 0.6'dan 0.4'e düşürdük
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Sadece kişi sınıfı (class_id = 0)
                    if class_id == 0 and confidence > confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        logger.info(f"Person detected with confidence: {confidence:.2f}")
            
            # NMS parametrelerini ayarla
            nms_threshold = 0.3  # 0.4'ten 0.3'e düşürdük
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            # Tespit kutularını çiz
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]
                    
                    # Kutuyu çiz
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    # Güven skorunu göster
                    text = f'Person: {confidence:.2f}'
                    cv2.putText(frame, 
                              text,
                              (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8,           # Font boyutu
                              (0, 255, 0),   # Yeşil renk
                              2)             # Kalınlık
                    
                    logger.info(f"Drew detection box at ({x}, {y}) with confidence {confidence:.2f}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in detect_persons: {str(e)}")
            logger.error("Error details:", exc_info=True)
            return frame

    def create_gstreamer_pipeline(self):
        GSTREAMER_PATH = "D:\\gstreamer\\1.0\\msvc_x86_64\\bin\\gst-launch-1.0.exe"
        
        return [
            GSTREAMER_PATH,
            "mfvideosrc",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420,width=640,height=480",
            "!",
            "x264enc",
            "tune=zerolatency",
            "speed-preset=ultrafast",
            "bitrate=2000",
            "!",
            "h264parse",
            "!",
            "rtph264pay",
            "!",
            "udpsink",
            "host=127.0.0.1",
            "port=5000",
            "sync=false"
        ]

    def start_stream(self):
        if self.running:
            return False
            
        self.running = True
        self.cap = cv2.VideoCapture(0)  # Webcam'i aç
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.thread = threading.Thread(target=self._stream_video)
        self.thread.start()
        return True

    def stop_stream(self):
        if not self.running:
            return False
            
        self.running = False
        if self.process:
            self.process.terminate()
        if self.cap:
            self.cap.release()
        if self.thread:
            self.thread.join()
        return True

    def _stream_video(self):
        try:
            # OpenCV ile kamerayı aç
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                logger.error("Cannot open camera")
                return
            
            logger.info("Camera opened successfully")
            
            # GStreamer pipeline'ı başlat
            pipeline_cmd = self.create_gstreamer_pipeline()
            logger.info(f"Starting GStreamer with command: {' '.join(pipeline_cmd)}")
            
            self.process = subprocess.Popen(
                pipeline_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Can't receive frame")
                    continue
                    
                # YOLO ile kişi tespiti yap
                if self.enable_detection:
                    try:
                        frame = self.detect_persons(frame)
                    except Exception as e:
                        logger.error(f"YOLO detection error: {str(e)}")
                
                # Frame'i göster
                cv2.imshow('YOLO Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            logger.error("Error details:", exc_info=True)
        finally:
            if self.process:
                self.process.terminate()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Stream stopped")

stream = VideoStream()

@app.route('/start', methods=['POST'])
def start_stream():
    if stream.start_stream():
        return jsonify({"message": "Stream started"}), 200
    return jsonify({"message": "Stream already running"}), 400

@app.route('/stop', methods=['POST'])
def stop_stream():
    if stream.stop_stream():
        return jsonify({"message": "Stream stopped"}), 200
    return jsonify({"message": "No stream running"}), 400

@app.route('/detection', methods=['POST'])
def toggle_detection():
    data = request.json
    if 'enable' in data:
        stream.enable_detection = data['enable']
        return jsonify({
            "message": f"Person detection {'enabled' if data['enable'] else 'disabled'}"
        }), 200
    return jsonify({"message": "Invalid request"}), 400

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "streaming": stream.running,
        "detection_enabled": stream.enable_detection,
        "compression_level": stream.compression_level,
        "resolution": stream.resolution,
        "framerate": stream.framerate
    }), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
