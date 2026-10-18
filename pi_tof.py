"""
Eigen Pi ToF Presence Detector (Milestone B trigger)

Reads the 8x8 VL53L7CX ToF depth sensor over I2C and detects whether anything
is within a threshold distance. This is Eigen's always-on "something is near
me" sense — cheap enough to poll constantly, and it wakes the camera pipeline
when presence is detected.

Hardware: 8x8 ToF (VL53L7CX + RP2040) on I2C bus 1, address 0x29 or 0x33.

Dependencies (on the Pi):
    pip install sparkfun-vl53l7cx   # or the vendor's python package

Usage:
    python pi_tof.py           # print presence + nearest distance each tick
"""

import os
import time

# Adjust to your sensor's I2C address (0x29 default for VL53L7CX; many
# SparkFun RP2040 boards expose 0x33).
I2C_ADDR = int(os.environ.get("EIGEN_TOF_ADDR", "0x33"), 16)
PRESENCE_DIST_MM = 1500      # "near" threshold (1.5 m)
POLL_INTERVAL = 0.2          # seconds


def make_sensor():
    """Create the VL53L7CX sensor object, tolerating library import failure."""
    try:
        import vl53l7cx
    except ImportError:
        print("[tof] vl53l7cx library not installed. Run: "
              "pip install sparkfun-vl53l7cx", flush=True)
        return None
    try:
        sensor = vl53l7cx.VL53L7CX(bus_id=1, address=I2C_ADDR)
        sensor.init()
        sensor.set_ranging_frequency_hz(15)
        sensor.start_ranging()
        return sensor
    except Exception as e:
        print(f"[tof] init failed: {e}", flush=True)
        return None


def read_grid(sensor):
    """Return the 8x8 distances in mm, or None on error."""
    try:
        # The sparkfun lib exposes get_ranging_data() / get_8x8 distance grid.
        if hasattr(sensor, "get_ranging_data"):
            grid = sensor.get_ranging_data()
        else:
            return None
        # Normalize to a flat list; format varies by library version.
        if isinstance(grid, dict):
            dist = grid.get("distance_mm")
        else:
            dist = grid
        return dist
    except Exception:
        return None


def nearest_and_presence(dist_grid):
    """From the 8x8 grid, return (nearest_mm, something_near)."""
    try:
        import numpy as np
        arr = np.asarray(dist_grid, dtype=float)
        # VL53L7CX uses 0 for invalid/no-target readings.
        valid = arr[(arr > 0)]
        nearest = float(valid.min()) if valid.size else None
        near = nearest is not None and nearest <= PRESENCE_DIST_MM
        return nearest, near
    except Exception:
        return None, False


def main():
    sensor = make_sensor()
    if sensor is None:
        return
    print(f"[tof] sensor running at 0x{I2C_ADDR:02x}, presence threshold "
          f"{PRESENCE_DIST_MM}mm", flush=True)
    print("[tof] press Ctrl+C to stop.", flush=True)
    try:
        while True:
            grid = read_grid(sensor)
            nearest, near = nearest_and_presence(grid)
            if nearest is not None:
                status = "NEAR" if near else "clear"
                print(f"[tof] nearest {nearest:.0f}mm [{status}]",
                      end="\r", flush=True)
                # Presence detected -> this is where we'd wake the camera.
                if near:
                    print("\n[tof] presence detected — (camera would wake here)",
                          flush=True)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[tof] stopping.", flush=True)
    finally:
        try:
            sensor.stop_ranging()
        except Exception:
            pass


if __name__ == "__main__":
    main()
