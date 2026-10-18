"""
Eigen Vision Receiver (Dell side) — tiered vision

Two vision tiers so Eigen can "react fast" AND "look closely" without a heavy
model always resident:

  /capture   -> Moondream2 (1.8B, ~0.5s)  : quick "what's there?" for reactions
  /inspect   -> qwen2.5vl:7b (~9s)        : deep look, read text, understand detail

Both log into Eigen's episodic + semantic memory. The Pi's motion/ToF trigger
hits /capture for instant feedback; deeper questions hit /inspect.

Run: python vision_receive.py  (systemd: eigen-vision.service)
"""

import io
import base64
import sqlite3
from datetime import datetime

import ollama
from flask import Flask, request, jsonify

import memory

FAST_MODEL = "moondream"       # quick reaction tier (when latency > accuracy)
DEEP_MODEL = "qwen2.5vl:7b"    # primary vision: deep understanding (default)
DB_PATH = "/home/eigen/eigen_memory.db"
MAX_DIM = 1024

FAST_PROMPT = (
    "Describe this image in one short single sentence. Just the sentence, "
    "nothing else."
)

DEEP_PROMPT = (
    "You are Eigen, an embodied AI at home. Describe what you can see in this "
    "image in 1-3 short, natural sentences, from a first-person perspective "
    "('I can see...'). Mention people, pets, objects, and the setting. Stay "
    "factual; don't invent things you can't actually see."
)

app = Flask(__name__)


def _downscale(image_bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    w, h = img.size
    scale = MAX_DIM / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _run(model, image_bytes, prompt, num_ctx):
    b64 = base64.b64encode(_downscale(image_bytes)).decode("utf-8")
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
        options={"num_predict": 300, "num_ctx": num_ctx},
    )
    return resp.message.content or ""


def _log(description, tier):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO episodic_memory (timestamp, role, event_type, content) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), "eigen", f"vision_{tier}", description),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
    try:
        memory.remember_memory(f"I saw: {description}", metadata={"source": f"vision_{tier}"})
    except Exception:
        pass


@app.route("/capture", methods=["POST"])
def capture():
    """Primary vision endpoint — uses the DEEP 7B model for accurate seeing."""
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    image_bytes = request.files["image"].read()
    prompt = (request.form.get("prompt") or "").strip() or DEEP_PROMPT
    try:
        description = _run(DEEP_MODEL, image_bytes, prompt, num_ctx=8192)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    _log(description, "deep")
    return jsonify({"description": description, "tier": "deep"})


@app.route("/inspect", methods=["POST"])
def inspect():
    """Alias for /capture (deep look). Kept for clarity."""
    return capture()


@app.route("/quick", methods=["POST"])
def quick():
    """Fast reaction tier — Moondream2, when speed matters more than detail."""
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    image_bytes = request.files["image"].read()
    prompt = (request.form.get("prompt") or "").strip() or FAST_PROMPT
    try:
        description = _run(FAST_MODEL, image_bytes, prompt, num_ctx=1024)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    _log(description, "fast")
    return jsonify({"description": description, "tier": "fast"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"ok": True, "capture": DEEP_MODEL, "quick": FAST_MODEL})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
