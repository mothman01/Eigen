"""
Eigen Pi Orchestrator (Milestone A + B combined)

Single bring-up entry point for the Pi. Ties the two senses together:

  * ToF (VL53L7CX) — cheap always-on presence detection.
  * Camera (OV5647) — visual capture, motion-gated.

Behavior (event-driven attention):
  1. Poll the ToF constantly. When something comes within threshold, OR the
     camera sees meaningful motion, wake the visual pipeline.
  2. Grab a frame and POST it to the Dell's vision receiver (qwen2.5vl).
  3. Print Eigen's first-person description.

This avoids streaming idle frames — the VLM only runs on significant events.

Usage:
    python pi_main.py

Set EIGEN_DELL_URL to override the Dell address. ToF and camera are both
optional: if either fails to init, the script degrades gracefully to the other.
"""

import os
import time

import cv2
import requests

DELL_URL = os.environ.get("EIGEN_DELL_URL", "http://192.168.1.32:5050")
SEND_WIDTH = 512
MOTION_THRESHOLD = 5.0
SEND_COOLDOWN = 4.0
PRESENCE_DIST_MM = 1500
POLL_INTERVAL = 0.25

# --- optional import blocks (so one failing sensor doesn't crash the other) ---
picamera2 = None
vl53l7cx = None
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None
try:
    import vl53l7cx  # noqa
except Exception:
    vl53l7cx = None


def init_camera():
    if Picamera2 is None:
        print("[main] picamera2 unavailable — running ToF-only.", flush=True)
        return None
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (1296, 972)}))
    cam.start()
    return cam


def init_tof():
    if vl53l7cx is None:
        print("[main] vl53l7cx unavailable — running camera-only.", flush=True)
        return None
    try:
        import vl53l7cx as v
        sensor = v.VL53L7CX(bus_id=1, address=0x33)
        sensor.init()
        sensor.set_ranging_frequency_hz(15)
        sensor.start_ranging()
        return sensor
    except Exception as e:
        print(f"[main] ToF init failed ({e}) — camera-only.", flush=True)
        return None


def tof_presence(sensor):
    try:
        grid = sensor.get_ranging_data()
        if isinstance(grid, dict):
            grid = grid.get("distance_mm")
        import numpy as np
        arr = np.asarray(grid, dtype=float)
        valid = arr[arr > 0]
        nearest = float(valid.min()) if valid.size else None
        return (nearest is not None and nearest <= PRESENCE_DIST_MM), nearest
    except Exception:
        return False, None


def send_frame(frame, reason):
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = frame.shape[:2]
    scale = SEND_WIDTH / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return
    print(f"[main] {reason} -> sending frame", flush=True)
    try:
        r = requests.post(f"{DELL_URL}/capture",
                          files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                          timeout=15)
        desc = r.json().get("description", "(no reply)")
        print(f"[main] Eigen: {desc}", flush=True)
    except Exception as e:
        print(f"[main] send failed: {e}", flush=True)


def main():
    cam = init_camera()
    tof = init_tof()
    if cam is None and tof is None:
        print("[main] no sensors available. Exiting.", flush=True)
        return

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=36)
    last_sent = 0.0
    first_frame = cam is not None

    print(f"[main] running. camera={cam is not None}, tof={tof is not None}", flush=True)
    print("[main] press Ctrl+C to stop.", flush=True)

    try:
        while True:
            wake = False
            reason = None

            if tof is not None:
                near, nearest = tof_presence(tof)
                if near:
                    wake = True
                    reason = f"ToF presence {nearest:.0f}mm"

            frame = None
            if cam is not None:
                frame = cam.capture_array()
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) \
                        if frame.shape[2] == 4 else frame
                    g = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    g = cv2.GaussianBlur(g, (21, 21), 0)
                    fg = bg_sub.apply(g)
                    mfrac = (cv2.countNonZero(fg) / float(fg.size)) * 100.0
                    if mfrac >= MOTION_THRESHOLD and not wake:
                        wake = True
                        reason = f"motion {mfrac:.1f}%"

            now = time.time()
            if first_frame and (now - last_sent >= SEND_COOLDOWN):
                send_frame(frame, "startup frame")
                last_sent = now
                first_frame = frame is None
            elif wake and (now - last_sent) >= SEND_COOLDOWN:
                send_frame(frame, reason)
                last_sent = now

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[main] stopping.", flush=True)
    finally:
        if cam is not None:
            cam.stop()
        if tof is not None:
            try:
                tof.stop_ranging()
            except Exception:
                pass


if __name__ == "__main__":
    main()
