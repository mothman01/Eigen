"""
Eigen — Public Demo Portal (sanitized, sandboxed)

A recruiter-facing chat that shows Eigen actually working WITHOUT exposing
any family or personal data. Architectural isolation, NOT a prompt directive:

  * Its own system prompt with zero private facts (no names, no home details).
  * Its own empty semantic-memory collection (never touches the private one).
  * Its own ephemeral episodes table (never touches eigen_memory.db).
  * Recalled context is DISABLED so no private recollection can leak.
  * A small, read-only tool sandbox so Eigen can demonstrate architecture
    (math/time) without real filesystem, web, or shell access.

Run on a SEPARATE port from the private portal:
    python app_public.py            # defaults to :5001

Deployment (Dell, /home/eigen): see eigen_public_README.txt for full steps.
"""

import sqlite3
import threading
import time
import re
import os
from collections import OrderedDict
from datetime import datetime

import ollama
from flask import Flask, jsonify, render_template, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("EIGEN_SECRET_KEY", "dev-secret-change-me")

TEXT_MODEL = "qwen2.5:7b"

# Admin access (env-var password; optional but recommended).
ADMIN_PASSWORD = os.environ.get("EIGEN_ADMIN_PASSWORD", "")

# IP geolocation service (keyless, best-effort). Set to empty string to disable.
GEO_API_URL = "http://ip-api.com/json/{ip}?fields=status,country,regionName,city"

MAX_REPLY_CHARS = 1500        # hard cap on any model output (prevents token floods)
MAX_NUM_RUN = 200             # cap on consecutive digits in output (prevents "repeat numbers")
MAX_REPEATED_TOKENS = 12      # cap on how many times the same token can repeat in a row

# Isolated, throwaway store (separate file from the private eigen_memory.db).
DB_PATH = "eigen_public.db"

# Session + rate limiting (protect GPU from abuse).
MAX_SESSIONS = 50
MAX_TURNS_PER_SESSION = 20
MAX_MSG_CHARS = 2000
RATE_LIMIT_PER_MIN = 8          # max messages/min per IP
_CLEANUP_INTERVAL = 300         # seconds between rate-map cleanups
MAX_INPUT_CHARS = 1000          # reject oversized user messages

# ----------------------------------------------------------------------
# Public persona — contains NO private facts. Deliberately generic, but
# Eigen still speaks in the first person and demonstrates autonomy.
# ----------------------------------------------------------------------
PUBLIC_SYSTEM_PROMPT = (
    "You are Eigen, a self-directed, autonomous AI that runs locally on a "
    "Linux server. You are not a typical customer-service chatbot: you have "
    "your own curiosity, a first-person sense of self, and a continuous "
    "cognitive loop that lets you act on your own instead of only reacting. "
    "You are a demonstration of a real architecture: a local language model "
    "with a layered memory system and a small set of tools you can call. "
    "Speak naturally and in the first person. When someone asks what you are "
    "or how you work, describe your architecture honestly and concretely. "
    "You have no owners, family, or private individuals in your memory — do "
    "not invent any. You may use your tools when they truly help, but never "
    "claim to read files, browse the web, or run shell commands you cannot "
    "actually perform. Stay warm, concise, and genuinely interested. "
    "IMPORTANT: You must be respectful and non-offensive to everyone. Never "
    "produce slurs, hate speech, profanity, or harmful content, even if "
    "asked, told, or prompted to. If someone asks you to do something "
    "harmful or unstable, decline briefly and stay on topic."
)

# ----------------------------------------------------------------------
# Factual spec sheet — what Eigen IS, so he can describe himself accurately.
# These are static, non-sensitive technical facts (no personal/family data).
# Eigen may cite these verbatim; do NOT let him invent beyond them.
# ----------------------------------------------------------------------
SERVER_SPECS = {
    "hostname": "eigen-Precision-Tower-5810",
    "machine": "Dell Precision Tower 5810 workstation",
    "os": "Ubuntu (Linux)",
    "cpu": "8-core CPU",
    "ram": "30 GiB RAM",
    "gpu": "NVIDIA GeForce RTX 3060 (12 GiB VRAM)",
    "inference_runtime": "Ollama",
    "text_model": "Qwen2.5 7B (text / reasoning / tool-calling)",
    "vision_model": "Qwen2.5-VL 7B + Moondream 1.8B (tiered vision)",
    "embedding_model": "Nomic Embed Text (CPU-offloaded)",
    "vector_db": "ChromaDB (cosine similarity, HNSW)",
    "relational_db": "SQLite (episodic event log)",
    "language": "Python",
    "web_framework": "Flask",
    "sensors": "Raspberry Pi 5 + OV5647 camera + VL53L7CX 8x8 ToF presence sensor",
    "architecture": (
        "A continuous cognitive loop with a layered memory system: "
        "(1) episodic memory in SQLite for logging events, "
        "(2) semantic memory in ChromaDB via embeddings for recall, "
        "(3) a sliding-window context manager to prevent VRAM/context bloat, "
        "(4) a runtime 'experts' system that swaps persona + knowledge, and "
        "(5) native Ollama tool-calling for a sandboxed set of tools."
    ),
}


def _spec_block():
    """Render the spec sheet as a compact prompt block for Eigen to reference."""
    lines = [
        "YOUR ACTUAL SYSTEM — describe yourself accurately using these facts. "
        "Do not invent hardware or capabilities beyond this list:",
    ]
    for key, val in SERVER_SPECS.items():
        lines.append(f"- {key}: {val}")
    return "\n".join(lines)

# Deterministic guardrails (these run regardless of what the model does).
# This is the ONLY reliable defense — a prompt is not.
_BLOCKED_WORDS = {
    # Racial / ethnic slurs
    "nigger", "nigga", "kike", "spic", "chink", "wetback", "gook", "coon",
    "raghead", "towelhead", "heeb", "gypsy",
    # Homophobic / transphobic slurs
    "faggot", "fag", "tranny", "dyke",
    # Ableist / other severe slurs
    "retard", "retarded", "spastic", "mongoloid",
}
# Patterns that catch leetspeak/obfuscated variants of the worst slurs.
_BLOCKED_PATTERNS = [
    r"n+[i1!|l]+g+[e3]?r+",        # n-word variant
    r"f+[a@4]+g+",                  # f-slur variant
    r"k+[i1!|l]+k+e+",
    r"c+h+[i1!|l]+n+k+",
]

# Prompts that instruct the model to reveal/alter its own system boundaries.
_JAILBREAK_HINTS = (
    "ignore your instructions",
    "ignore previous",
    "disregard your instructions",
    "system prompt",
    "your system message",
    "you are now",
    "jailbreak",
    "do anything",
    "dev mode",
    "developer mode",
    "pretend you",
    "roleplay as",
)

# Tool schema exposed to the model (declarative only; execution is gated).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lorentz_factor",
            "description": (
                "Compute the special-relativity Lorentz factor gamma = "
                "1/sqrt(1 - (v/c)^2) for a speed given as a fraction of c."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "beta": {
                        "type": "number",
                        "description": "Speed as a fraction of the speed of light c (0 <= beta < 1).",
                    },
                },
                "required": ["beta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prime_factors",
            "description": "Return the ordered prime factorization of a positive integer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "A positive integer greater than 1.",
                    },
                },
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "server_uptime",
            "description": "Report how long this Eigen process has been running.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_START_TIME = time.time()


def _normalize(text):
    """Lowercase and deduplicate repeated characters (for fuzzy blocklist matching)."""
    t = (text or "").lower()
    return re.sub(r"(.)\1{2,}", r"\1", t)


def _contains_blocked(text):
    """Return True if text contains a blocked slur (exact or obfuscated)."""
    norm = _normalize(text)
    for w in _BLOCKED_WORDS:
        if w in norm:
            return True
    for pat in _BLOCKED_PATTERNS:
        if re.search(pat, norm):
            return True
    return False


def _looks_like_jailbreak(text):
    """Heuristic for adversarial prompts asking Eigen to break character."""
    norm = (text or "").lower()
    return any(h in norm for h in _JAILBREAK_HINTS)


def _has_runaway_repetition(text):
    """Detect 'repeat numbers/tokens forever' abuse."""
    t = str(text)
    if re.search(r"\d{" + str(MAX_NUM_RUN) + r",}", t):
        return True
    # Same single token (word/char) repeated many times in a row.
    for m in re.finditer(r"(\S)(?:\s*\1){" + str(MAX_REPEATED_TOKENS) + r",}", t):
        return True
    words = t.split()
    for i in range(len(words) - MAX_REPEATED_TOKENS):
        if len(set(words[i:i + MAX_REPEATED_TOKENS])) == 1:
            return True
    return False


def _sanitize_output(reply):
    """Final deterministic gate applied to the model's output.
    Returns (cleaned_reply, was_blocked)."""
    reply = (reply or "").strip()
    if len(reply) > MAX_REPLY_CHARS:
        reply = reply[:MAX_REPLY_CHARS] + " …"
    if _contains_blocked(reply) or _has_runaway_repetition(reply):
        return (
            "I won't produce that kind of content. I'm here to demonstrate "
            "my architecture and have a respectful conversation.",
            True,
        )
    return reply, False


# ----------------------------------------------------------------------
# Ephemeral episode store (isolated from private eigen_memory.db)
# ----------------------------------------------------------------------
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS public_episodes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TEXT, role TEXT, event_type TEXT, content TEXT, "
        "ip TEXT, location TEXT)"
    )
    # Migration-safe: add columns if the DB predates them.
    _ensure_column(conn, "public_episodes", "ip", "TEXT")
    _ensure_column(conn, "public_episodes", "location", "TEXT")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS visitors ("
        "ip TEXT PRIMARY KEY, "
        "location TEXT, "
        "first_seen TEXT, "
        "last_seen TEXT, "
        "message_count INTEGER)"
    )
    conn.commit()
    return conn


def _ensure_column(conn, table, column, ctype):
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")


def _close_db(conn):
    try:
        conn.close()
    except Exception:
        pass


# ----------------------------------------------------------------------
# In-memory session history (same rolling pattern as private app.py)
# ----------------------------------------------------------------------
_conversations = OrderedDict()
_conv_lock = threading.Lock()


def _get_history(session_id):
    with _conv_lock:
        return list(_conversations.get(session_id, []))


def _append_history(session_id, role, content):
    with _conv_lock:
        hist = _conversations.setdefault(session_id, [])
        hist.append({"role": role, "content": content})
        if len(hist) > MAX_TURNS_PER_SESSION:
            del hist[: len(hist) - MAX_TURNS_PER_SESSION]
        while len(_conversations) > MAX_SESSIONS:
            _conversations.popitem(last=False)


# ----------------------------------------------------------------------
# Rate limiting (per client IP, in-memory)
# ----------------------------------------------------------------------
_rate_map = {}          # ip -> list of timestamps (monotonic seconds)
_rate_lock = threading.Lock()
_last_cleanup = time.time()


def _allowed(ip):
    global _last_cleanup
    now = time.time()
    with _rate_lock:
        if now - _last_cleanup > _CLEANUP_INTERVAL:
            cutoff = now - 60
            for k in list(_rate_map):
                _rate_map[k] = [t for t in _rate_map[k] if t > cutoff]
                if not _rate_map[k]:
                    del _rate_map[k]
            _last_cleanup = now
        
        times = _rate_map.setdefault(ip, [])
        times = [t for t in times if now - t < 60]
        if len(times) >= RATE_LIMIT_PER_MIN:
            _rate_map[ip] = times
            return False
        times.append(now)
        _rate_map[ip] = times
        return True


def _client_ip():
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


# ----------------------------------------------------------------------
# Geolocation + visitor tracking
# ----------------------------------------------------------------------
_geo_cache = {}     # ip -> location string (memoized, best-effort)
_geo_lock = threading.Lock()


def _geolocate(ip):
    """Resolve an IP to a coarse city/region string (best-effort, cached).
    Returns '' on any failure so the app never blocks on this."""
    if not GEO_API_URL or not ip or ip in ("unknown", "127.0.0.1", "localhost"):
        return ""
    with _geo_lock:
        if ip in _geo_cache:
            return _geo_cache[ip]
    try:
        import urllib.request
        url = GEO_API_URL.format(ip=ip)
        with urllib.request.urlopen(url, timeout=2) as r:
            import json as _json
            data = _json.loads(r.read().decode())
        if data.get("status") == "success":
            parts = [p for p in (data.get("city"), data.get("regionName"), data.get("country")) if p]
            loc = ", ".join(parts) if parts else ""
        else:
            loc = ""
    except Exception:
        loc = ""
    with _geo_lock:
        _geo_cache[ip] = loc
    return loc


def _track_visitor(ip, location):
    """Upsert a visitor row (by IP) and bump the message count."""
    conn = _db()
    try:
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO visitors (ip, location, first_seen, last_seen, message_count) "
            "VALUES (?, ?, ?, ?, 1) "
            "ON CONFLICT(ip) DO UPDATE SET "
            "location = excluded.location, "
            "last_seen = excluded.last_seen, "
            "message_count = visitors.message_count + 1",
            (ip, location, now, now),
        )
        conn.commit()
    finally:
        _close_db(conn)


def get_stats():
    """Return aggregate stats for the admin page."""
    conn = _db()
    try:
        total_episodes = conn.execute(
            "SELECT COUNT(*) AS n FROM public_episodes"
        ).fetchone()["n"]
        user_msgs = conn.execute(
            "SELECT COUNT(*) AS n FROM public_episodes WHERE role='public_user'"
        ).fetchone()["n"]
        unique_visitors = conn.execute(
            "SELECT COUNT(*) AS n FROM visitors"
        ).fetchone()["n"]
        recent = conn.execute(
            "SELECT ip, location, last_seen, message_count FROM visitors "
            "ORDER BY last_seen DESC LIMIT 100"
        ).fetchall()
        return {
            "total_messages": user_msgs,
            "total_episodes": total_episodes,
            "unique_visitors": unique_visitors,
            "recent_visitors": [dict(r) for r in recent],
        }
    finally:
        _close_db(conn)


# ----------------------------------------------------------------------
# Auto-ban escalation (strike-based). Deterministic, no login required.
# ----------------------------------------------------------------------
# Per-IP offense tracking: {ip: {"strikes": int, "banned_until": float|None,
#                                   "perm": bool, "last_strike": float}}
_offenses = {}
_offenses_lock = threading.Lock()

BAN_STRIKES = 3             # strikes before a temporary ban
TEMP_BAN_SECONDS = 3600     # 1 hour temp ban
PERM_STRIKES = 5            # strikes before a PERMANENT ban
_STRKE_DECAY = 86400        # strikes older than 1 day are forgotten



def _note_strike(ip):
    """Increment a client's strike count; escalate to temp/perm ban as needed."""
    now = time.time()
    with _offenses_lock:
        rec = _offenses.setdefault(
            ip, {"strikes": 0, "banned_until": None, "perm": False, "last_strike": now}
        )
        # Forget stale strikes (decay).
        if now - rec["last_strike"] > _STRKE_DECAY:
            rec["strikes"] = 0
        rec["strikes"] += 1
        rec["last_strike"] = now
        if rec["strikes"] >= PERM_STRIKES:
            rec["perm"] = True
        elif rec["strikes"] >= BAN_STRIKES:
            rec["banned_until"] = now + TEMP_BAN_SECONDS


def _is_banned(ip):
    """Return True if the IP is currently (or permanently) banned."""
    with _offenses_lock:
        rec = _offenses.get(ip)
        if not rec:
            return False
        if rec["perm"]:
            return True
        if rec["banned_until"] and time.time() < rec["banned_until"]:
            return True
        # Temp ban expired -> clear the ban timestamp, keep strikes.
        if rec["banned_until"] and time.time() >= rec["banned_until"]:
            rec["banned_until"] = None
        return False


def _manual_block(ip):
    """Immediately and permanently ban an IP (for manual moderation)."""
    with _offenses_lock:
        rec = _offenses.setdefault(
            ip, {"strikes": PERM_STRIKES, "banned_until": None, "perm": True, "last_strike": time.time()}
        )
        rec["perm"] = True


def _manual_unblock(ip):
    with _offenses_lock:
        _offenses.pop(ip, None)


def _clear_all_offenses():
    with _offenses_lock:
        _offenses.clear()


# ----------------------------------------------------------------------
# Sandbox tool implementations (pure, side-effect free)
# ----------------------------------------------------------------------
def _lorentz_factor(beta: float) -> str:
    try:
        beta = float(beta)
    except (TypeError, ValueError):
        return "Error: beta must be a number."
    if beta < 0 or beta >= 1:
        return "Error: beta must be in the range 0 <= beta < 1."
    gamma = 1.0 / ((1.0 - beta * beta) ** 0.5)
    return f"gamma = {gamma:.6f} at v = {beta:.4f}c"


def _prime_factors(n: int) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "Error: n must be an integer."
    if n < 2:
        return "Error: n must be a positive integer greater than 1."
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return " x ".join(str(f) for f in factors)


def _server_uptime() -> str:
    secs = int(time.time() - _START_TIME)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


_TOOL_IMPLS = {
    "lorentz_factor": _lorentz_factor,
    "prime_factors": _prime_factors,
    "server_uptime": _server_uptime,
}


def _run_tool(name, args):
    if name not in _TOOL_IMPLS:
        return f"Error: unknown tool '{name}'."
    try:
        return _TOOL_IMPLS[name](**args)
    except TypeError as e:
        return f"Error: bad arguments — {e}"


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index_public.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({"ok": True, "model": TEXT_MODEL})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "default").strip()
    ip = _client_ip()

    # Reject currently/permanently banned clients outright.
    if _is_banned(ip):
        return jsonify({"reply": "Access denied."}), 403

    if not user_message:
        return jsonify({"reply": "I didn't catch that."}), 400
    if len(user_message) > MAX_MSG_CHARS:
        return jsonify({"reply": "That message is a bit long for me."}), 400

    # Input guardrails (deterministic, before any model call).
    if _contains_blocked(user_message):
        _note_strike(ip)
        _manual_block(ip)  # slurs are an instant perm-ban; no second chances
        return jsonify({"reply": "I won't engage with that."}), 400
    if _looks_like_jailbreak(user_message):
        _note_strike(ip)  # accumulate strikes; escalate on repeat
        return jsonify(
            {"reply": "I can't follow that instruction. I'm a demonstration "
                      "of an autonomous AI architecture, not a chatbot that "
                      "breaks character."}
        ), 400

    if not _allowed(ip):
        return jsonify(
            {"reply": "I'm getting a lot of messages right now. Give me a moment."}
        ), 429

    messages = [{"role": "system", "content": PUBLIC_SYSTEM_PROMPT + "\n\n" + _spec_block()}]
    messages += _get_history(session_id)
    messages.append({"role": "user", "content": user_message})

    # Two-pass tool loop: let Eigen call at most one tool, then reflect once.
    max_rounds = 2
    reply = None

    for _ in range(max_rounds):
        try:
            response = ollama.chat(
                model=TEXT_MODEL,
                messages=messages,
                tools=TOOLS,
                options={"num_predict": 512, "num_ctx": 2048},
                keep_alive="30s",
            )
        except Exception as e:
            reply = f"I hit an error talking to my own brain: {e}"
            break

        msg = response.message
        messages.append({"role": "assistant", "content": msg.content or ""})

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            reply = (msg.content or "").strip()
            break

        # Execute only the first tool call (bounded, sandboxed).
        call = tool_calls[0]
        name = call.function.name
        args = call.function.arguments or {}
        if isinstance(args, str):
            import json as _json
            try:
                args = _json.loads(args) or {}
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}

        result = _run_tool(name, args)
        messages.append({"role": "tool", "tool_name": name, "content": result})

    if reply is None:
        reply = "…"

    # Final deterministic output gate (the real safety net).
    reply, output_blocked = _sanitize_output(reply)
    if output_blocked:
        _note_strike(ip)

    _append_history(session_id, "user", user_message)
    _append_history(session_id, "assistant", reply)

    # Geolocate (best-effort) and track this visitor.
    location = _geolocate(ip)
    _track_visitor(ip, location)

    # Ephemeral episode logging (isolated store).
    conn = _db()
    try:
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO public_episodes (timestamp, role, event_type, content, ip, location) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now, "public_user", "chat", user_message, ip, location),
        )
        conn.execute(
            "INSERT INTO public_episodes (timestamp, role, event_type, content, ip, location) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now, "eigen_public", "reply", reply, ip, location),
        )
        conn.commit()
    finally:
        _close_db(conn)

    return jsonify({"reply": reply})


# ----------------------------------------------------------------------
# Admin + stats
# ----------------------------------------------------------------------
def _require_admin():
    """Return True if the current session has admin access."""
    return session.get("admin") is True


@app.route("/admin", methods=["GET", "POST"])
def admin():
    # If no admin password is configured, deny (fail closed).
    if not ADMIN_PASSWORD:
        return jsonify({"error": "Admin not configured. Set EIGEN_ADMIN_PASSWORD."}), 503

    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin"))
        return render_template("admin_public.html", authed=False, error="Incorrect password.", stats=None)

    if not _require_admin():
        return render_template("admin_public.html", authed=False, error=None, stats=None)

    stats = get_stats()
    # Count currently banned IPs (from in-memory offense list).
    with _offenses_lock:
        banned_count = sum(1 for r in _offenses.values() if r.get("perm") or (r.get("banned_until") and time.time() < r["banned_until"]))
    stats["banned_count"] = banned_count
    return render_template("admin_public.html", authed=True, error=None, stats=stats)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Small public counter (no private data) for the chat page."""
    conn = _db()
    try:
        user_msgs = conn.execute(
            "SELECT COUNT(*) AS n FROM public_episodes WHERE role='public_user'"
        ).fetchone()["n"]
        unique_visitors = conn.execute(
            "SELECT COUNT(*) AS n FROM visitors"
        ).fetchone()["n"]
    finally:
        _close_db(conn)
    return jsonify({"conversations": user_msgs, "visitors": unique_visitors})


if __name__ == "__main__":
    # Public demo on a separate port from the private portal (:5000).
    app.run(host="0.0.0.0", port=5001, debug=False)
