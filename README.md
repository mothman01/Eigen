# Eigen — Autonomous Local AI System

Eigen is a self-directed, autonomous AI that runs **fully locally** on a Dell
Precision Tower 5810 workstation. It is not a cloud API wrapper or a chatbot:
it is a continuous cognitive loop with layered memory, native tool-calling,
and multimodal senses (camera + time-of-flight presence detection) wired to a
Raspberry Pi 5.

**Live demo:** https://eigen.heigen.io

---

## Architecture

Eigen is built around a few core ideas:

```
┌───────────────┐     ┌──────────────────────────────┐
│ Raspberry Pi 5│────▶│ Dell Precision Tower 5810     │
│  camera + ToF │     │  Ubuntu + Ollama (local GPU)  │
└───────────────┘     │  ├─ episodic memory (SQLite)  │
                      │  ├─ semantic memory (ChromaDB)│
                      │  ├─ sliding-window context    │
                      │  ├─ runtime "experts" system  │
                      │  └─ sandboxed tool-calling    │
                      └──────────────────────────────┘
```

### Hardware

| Component | Detail |
|-----------|--------|
| Host | Dell Precision Tower 5810 |
| OS | Ubuntu (Linux) |
| CPU | 8-core |
| RAM | 30 GiB |
| GPU | NVIDIA GeForce RTX 3060 (12 GiB VRAM) |
| Sensing | Raspberry Pi 5 + OV5647 camera + VL53L7CX 8×8 ToF |

### Software stack

- **Inference:** [Ollama](https://ollama.com) — Qwen2.5 7B (text), Qwen2.5-VL 7B + Moondream 1.8B (tiered vision), Nomic Embed (CPU-offloaded embeddings)
- **Semantic memory:** ChromaDB (cosine similarity, HNSW)
- **Episodic memory:** SQLite
- **Web layer:** Flask
- **Language:** Python

### Layered memory model

1. **Episodic (SQLite)** — a chronological event log of everything Eigen does.
2. **Semantic (ChromaDB)** — vector embeddings enable contextual recall of past
   experiences rather than acting only on the latest message.
3. **Sliding-window context** — bounded context management prevents VRAM/context
   bloat over long autonomous runs.

### Tool-calling

Eigen uses native Ollama tool-calling with a **sandboxed, read-only toolset**
(math utilities and uptime) so it can demonstrate real agent behavior without
filesystem, shell, or web access.

### "Experts" system

Eigen can swap personas and domain knowledge at runtime (a prompt-level
mechanism, not weight modification) via a manifest of "experts."

---

## This repository

This repo contains the **public-facing and peripheral** components of Eigen.
The full autonomous loop, Discord integration, and personal-memory systems are
kept private.

### Files

| File | Purpose |
|------|---------|
| `app_public.py` | Sanitized, sandboxed public chat portal (runs on port 5001) |
| `templates/index_public.html` | Recruiter-facing chat UI |
| `vision_receive.py` | Tiered vision receiver (Moondream fast + Qwen2.5-VL deep) |
| `pi_main.py` | Raspberry Pi orchestrator (camera + ToF event-driven attention) |
| `pi_capture.py` | picamera2 vision capture with motion-delta gating |
| `pi_tof.py` | VL53L7CX time-of-flight presence detection |
| `web_history.py` | Audit log of web-fetch activity |
| `curriculum_runner.py` | Staged-learning gate (curriculum-gated intake) |
| `curriculum.json` | Curriculum stage definitions |
| `synthesize_theme.py` | Cross-referenced knowledge synthesis |
| `DEPLOYMENT.txt` | Full deploy + Cloudflare Tunnel runbook |

---

## Security & privacy design

Eigen's public demo is hardened through **architectural isolation**, not just
prompts:

- **Separate system prompt** with zero personal data.
- **Separate, empty memory store** (never touches private DBs).
- **Memory recall disabled** in public mode.
- **Read-only tool sandbox** (no shell/web/filesystem).
- **Deterministic output filters** — blocklist + jailbreak detection + runaway
  repetition detection.
- **Rate limiting** (per-IP) and **strike-based auto-ban escalation**.
- **Cloudflare in front** (Tunnel + Bot Fight Mode + WAF rate-limit rules).

---

## Local development

> **Note:** this runs against a local Ollama instance with the models listed
> above. You will need Ollama and the model weights installed to run it.

```bash
# 1. Install dependencies
pip install flask ollama

# 2. Pull the model
ollama pull qwen2.5:7b

# 3. Run the public portal
python app_public.py
# -> http://localhost:5001
```

---

## Deployment

See `DEPLOYMENT.txt` for the full production runbook (systemd services,
Cloudflare Tunnel, DNS, and access control).

---

## Built by

Michael Cogan — Electrical & Computer Engineering Technology student.
Resume: https://resume.heigen.io
