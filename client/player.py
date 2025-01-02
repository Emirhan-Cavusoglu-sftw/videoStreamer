import cv2

# GStreamer pipeline
gst_pipeline = 'udpsrc port=5000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink'
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Couldn't open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read video frame.")
        break
    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
