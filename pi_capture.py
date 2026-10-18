"""
Eigen Pi Vision Capture (Milestone A) — picamera2 backend

Runs on the Raspberry Pi 5 with the OV5647 CSI camera. Captures frames via
picamera2/libcamera (the correct backend for CSI cameras on Pi 5), applies
motion-delta gating, and POSTs keyframes to the Dell's vision receiver only
when something actually changes.

Dependencies:
    sudo apt install -y python3-picamera2 python3-venv python3-pip
    (in a venv) pip install opencv-python-headless requests
"""

import io
import os
import time

import cv2
import requests
from picamera2 import Picamera2

DELL_URL = os.environ.get("EIGEN_DELL_URL", "http://192.168.1.32:5050")
SEND_WIDTH = 512
MOTION_THRESHOLD = 5.0
SEND_COOLDOWN = 4.0
CHECK_INTERVAL = 0.3


def downscale(frame, max_dim=SEND_WIDTH):
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame


def encode_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else None


def send_frame(frame, reason):
    jpeg = encode_jpeg(frame)
    if not jpeg:
        return
    print(f"[pi] {reason} -> sending frame", flush=True)
    try:
        r = requests.post(
            f"{DELL_URL}/capture",
            files={"image": ("frame.jpg", jpeg, "image/jpeg")},
            timeout=15,
        )
        desc = r.json().get("description", "(no reply)")
        print(f"[pi] Eigen: {desc}", flush=True)
    except Exception as e:
        print(f"[pi] send failed: {e}", flush=True)


def main():
    picam = Picamera2()
    # Low-ish resolution preview stream for motion detection + keyframes.
    config = picam.create_preview_configuration(main={"size": (1296, 972)})
    picam.configure(config)
    picam.start()

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=36)
    last_sent = 0.0
    first_frame = True

    print(f"[pi] picamera2 started; keyframes -> {DELL_URL}/capture", flush=True)
    print("[pi] press Ctrl+C to stop.", flush=True)

    try:
        while True:
            frame = picam.capture_array()
            if frame is None:
                time.sleep(0.5)
                continue

            # picamera2 may return BGRA; convert to BGR.
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            frame = downscale(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            fg = bg_sub.apply(gray)
            motion_frac = (cv2.countNonZero(fg) / float(fg.size)) * 100.0

            print(f"[pi] motion {motion_frac:.1f}%", end="\r", flush=True)

            now = time.time()
            if first_frame:
                send_frame(frame, "startup frame")
                last_sent = now
                first_frame = False
            elif motion_frac >= MOTION_THRESHOLD and (now - last_sent) >= SEND_COOLDOWN:
                send_frame(frame, f"motion {motion_frac:.1f}%")
                last_sent = now

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n[pi] stopping.", flush=True)
    finally:
        picam.stop()


if __name__ == "__main__":
    main()
