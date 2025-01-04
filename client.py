import cv2
import subprocess
import logging
import signal
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    GSTREAMER_PATH = "D:\\gstreamer\\1.0\\msvc_x86_64\\bin\\gst-launch-1.0.exe"
    process = None
    
    pipeline_cmd = [
        GSTREAMER_PATH,
        "udpsrc",
        "port=5000",
        "!",
        "application/x-rtp,media=video,encoding-name=H264,payload=96",
        "!",
        "rtph264depay",
        "!",
        "h264parse",
        "!",
        "d3d11h264dec",
        "!",
        "d3d11videosink",
        "sync=false"
    ]
    
    try:
        logger.info("Starting GStreamer client...")
        cmd_str = " ".join(pipeline_cmd)
        logger.info(f"Command: {cmd_str}")
        
        process = subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        logger.info("Client started successfully")
        
        def signal_handler(sig, frame):
            logger.info("Stopping client...")
            if process:
                process.terminate()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        while True:
            output = process.stderr.readline()
            if output:
                logger.info(output.strip().decode())
            
            if process.poll() is not None:
                error = process.stderr.read().decode()
                logger.error(f"GStreamer error: {error}")
                break
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if process:
            process.terminate()
        logger.info("Client stopped")

if __name__ == "__main__":
    main()