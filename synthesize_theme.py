"""
Eigen Theme Synthesis (compiled topical knowledge)

After you've ingested several related textbooks into one expert (each under a
sub-topic), this script has Eigen write a UNIFIED, cross-referenced summary of
that whole theme — turning "a folder of separate notes" into one coherent,
integrated understanding (e.g., how physics, mechanical, and electrical
engineering connect).

Usage:
    python synthesize_theme.py engineer            # synthesize the 'engineer' expert
    python synthesize_theme.py engineer --list      # show sub-topics first

The output is written to experts/<expert>/_synthesis.md (loaded as core
context whenever Eigen is in that expert mode) and stored in semantic memory.
"""

import os
import sys

import ollama

import memory

TEXT_MODEL = "qwen2.5:7b"
BASE = "/home/eigen"

SYNTHESIS_PROMPT = (
    "You are Eigen, consolidating a themed body of specialist knowledge into a "
    "single integrated understanding. Below are study notes drawn from several "
    "books/topics within ONE field. Merge them into a coherent summary that:\n"
    "  - groups ideas into the major sub-areas\n"
    "  - explains how the sub-areas CONNECT (cross-links, shared principles)\n"
    "  - surfaces the key formulas/concepts a specialist must know\n"
    "  - is faithful (no invented facts)\n"
    "Write 20-40 numbered/bulleted lines. Be dense and useful, not fluffy. "
    "End with '---'.\n\n"
    "NOTES:\n"
)


def gather_notes(expert_id, max_chars=30000):
    kdir = os.path.join(BASE, "experts", expert_id)
    if not os.path.isdir(kdir):
        return "", []
    files = []
    for root, _, fnames in os.walk(kdir):
        for fn in fnames:
            if fn.endswith(".md") and not fn.startswith("_"):
                files.append(os.path.join(root, fn))
    text = ""
    topics = []
    for p in sorted(files):
        rel = os.path.relpath(p, kdir)
        topics.append(rel.split(os.sep)[0])
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text += f.read() + "\n"
    return text[-max_chars:], sorted(set(topics))


def main(expert_id):
    if "--list" in sys.argv:
        _, topics = gather_notes(expert_id)
        print("Sub-topics under", expert_id, ":", topics or "(none — ingest books first)")
        return

    text, topics = gather_notes(expert_id)
    if not text.strip():
        print(f"No notes found for expert '{expert_id}'. Ingest books first.")
        return

    print(f"Synthesizing '{expert_id}' (sub-topics: {topics})...")
    resp = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": SYNTHESIS_PROMPT + text}],
        options={"num_predict": 1500, "num_ctx": 8192},
    )
    body = (resp.message.content or "").split("---")[0].strip()

    out = os.path.join(BASE, "experts", expert_id, "_synthesis.md")
    with open(out, "w") as f:
        f.write(f"# {expert_id} — integrated understanding\n")
        f.write(body)

    for line in body.splitlines():
        line = line.strip().lstrip("-*•0123456789. ") .strip()
        if line and len(line) > 12:
            memory.remember_memory(line, metadata={"source": f"synthesis:{expert_id}"})

    print(f"Wrote synthesis to {out}")
    print(f"Total semantic memories: {memory.count_memories()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python synthesize_theme.py <expert_id> [--list]")
        sys.exit(1)
    main(sys.argv[1])
