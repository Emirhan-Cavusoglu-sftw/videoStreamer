from flask import Flask, request
import subprocess

app = Flask(__name__)
gst_process = None

@app.route('/start', methods=['POST'])
def start_stream():
    global gst_process
    if not gst_process:
        gst_process = subprocess.Popen(
            ["gst-launch-1.0", "autovideosrc", "!", "videoconvert", "!", "x264enc", "!", "rtph264pay", "!", "udpsink", "host=127.0.0.1", "port=5000"]
        )
        return "Streaming started", 200
    return "Stream already running", 400

@app.route('/stop', methods=['POST'])
def stop_stream():
    global gst_process
    if gst_process:
        gst_process.terminate()
        gst_process = None
        return "Streaming stopped", 200
    return "No stream running", 400

if __name__ == '__main__':
    app.run(debug=True)
