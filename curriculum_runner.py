"""
Eigen Curriculum Runner (staged intake — "A")

Implements strict staged learning. Eigen only INGESTS (studies) material that
belongs to the currently-active curriculum stage. Later stages' material is
held back until its stage becomes active.

The curriculum is defined in curriculum.json. Each stage lists which "experts"
(or domains) are being studied and how long it runs. This runner:
  1. Reports the active stage.
  2. Provides gate() so ingest scripts only accept data for the active stage's
     allowed experts.
  3. Provides advance() to move to the next stage when a week completes.

Usage:
    python curriculum_runner.py status        # show current stage
    python curriculum_runner.py advance       # move to next stage (manual)
"""

import json
import os
import sys
from datetime import datetime, timedelta

BASE = "/home/eigen"
CUR_PATH = os.path.join(BASE, "curriculum.json")
STATE_PATH = os.path.join(BASE, "curriculum_state.json")


def load_curriculum():
    with open(CUR_PATH, "r") as f:
        return json.load(f)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    # Default: start at stage 1, started today.
    return {"current_stage": 1, "started": datetime.now().isoformat()}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def active_stage():
    cur = load_curriculum()
    state = load_state()
    sid = state["current_stage"]
    for stage in cur["stages"]:
        if stage["id"] == sid:
            return stage
    return cur["stages"][0]


def allowed_experts():
    """Return the set of expert ids allowed for intake in the current stage."""
    return set(active_stage().get("experts", []))


def gate(expert_id):
    """Return True if this expert is allowed to be ingested right now."""
    return expert_id in allowed_experts()


def advance():
    cur = load_curriculum()
    state = load_state()
    nxt = state["current_stage"] + 1
    ids = [s["id"] for s in cur["stages"]]
    if nxt not in ids:
        print(f"Already at final stage ({state['current_stage']}).")
        return
    state["current_stage"] = nxt
    state["started"] = datetime.now().isoformat()
    save_state(state)
    print(f"Advanced to stage {nxt}: {cur['stages'][nxt-1]['name']}")


def status():
    stage = active_stage()
    started = load_state().get("started", "")
    print(f"Active stage {stage['id']}: {stage['name']}")
    print(f"  allowed experts: {stage.get('experts')}")
    print(f"  note: {stage.get('note')}")
    if started:
        print(f"  started: {started}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "advance":
        advance()
    else:
        print("Usage: python curriculum_runner.py [status|advance]")
