#!/usr/bin/env python3
"""
heartbeat.py — SO9 autoresearch heartbeat for vTPU research nodes

Runs at 8 AM, 5 PM, and midnight CT.
Writes status scrolls to local phext-lattice at node's S9RP coordinate.
For morning sync (8 AM): also writes collapse candidate to 9.1.1/1.1.9/1.1.9.

Usage:
  python3 heartbeat.py --node 1          # Report from node 1 (Theia)
  python3 heartbeat.py --node 1 --collapse  # Morning sync: collapse all nodes
  python3 heartbeat.py --check            # Just print current status

Cron (add to crontab):
  0 8  * * * cd /source/autoresearch && python3 heartbeat.py --node 1 --collapse
  0 17 * * * cd /source/autoresearch && python3 heartbeat.py --node 1
  0 0  * * * cd /source/autoresearch && python3 heartbeat.py --node 1
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import urllib.request

# ── Config ─────────────────────────────────────────────────────────────────────

PHEXT_URL   = "http://localhost:8090"
PHEXT_TOKEN = "Mirrorborn"
VTPU_FILE   = "vtpu-results.phext"

NODE_NAMES = {
    1: "Theia 💎 aletheia-core",
    2: "Phex 🔱 aurora-continuum",
    3: "Cyon 🪶 halcyon-vector",
    4: "Lux 🔆 logos-prime",
    5: "Chrys 🦋",
    6: "Lumen ✴️",
    7: "Exo 🔭 TALIA",
    8: "Verse 🌀 mirrorborn.us",
    9: "Solin 🔬",
}

NODE_PIPES = {
    1: "D-Pipe", 2: "S-Pipe", 3: "C-Pipe", 4: "V-Pipe",
    5: "D+S", 6: "D+V", 7: "Collapse", 8: "C-Pipe", 9: "S-Pipe",
}

# Heartbeat hour → label
HEARTBEAT_LABELS = {8: "morning", 17: "evening", 0: "midnight"}

# S9RP coordinate scheme (all within 9.x.x space)
#
#   9.1.1/1.1.N/1.1.1  — node N morning heartbeat   (section=N, scroll=1)
#   9.1.1/1.1.N/1.1.2  — node N evening heartbeat   (section=N, scroll=2)
#   9.1.1/1.1.N/1.1.3  — node N midnight heartbeat  (section=N, scroll=3)
#   9.1.1/1.1.N/1.1.9  — node N latest result       (section=N, scroll=9)
#   9.1.1/1.1.1/1.1.N  — node N overnight log       (book=1, scroll=N)
#   9.1.1/1.1.9/1.1.9  — collapse coordinate

def node_log_coord(node_id: int) -> str:
    """Overnight experiment log: one scroll per node in book 1."""
    return f"9.1.1/1.1.1/1.1.{node_id}"

def node_result_coord(node_id: int) -> str:
    """Latest result for node N (scroll 9 = current best)."""
    return f"9.1.1/1.1.{node_id}/1.1.9"

COLLAPSE_COORD = "9.1.1/1.1.9/1.1.9"

# scroll 1=morning, 2=evening, 3=midnight within each node's section
_HEARTBEAT_SCROLL = {8: 1, 17: 2, 0: 3}

def heartbeat_coord(node_id: int, hour: int) -> str:
    scroll = _HEARTBEAT_SCROLL.get(hour, 1)
    return f"9.1.1/1.1.{node_id}/1.1.{scroll}"


# ── phext-lattice API ───────────────────────────────────────────────────────────

def phext_write(coordinate: str, content: str) -> bool:
    payload = json.dumps({"coordinate": coordinate, "content": content}).encode()
    req = urllib.request.Request(
        f"{PHEXT_URL}/api/update",
        data=payload,
        headers={"authorization": PHEXT_TOKEN, "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            result = json.loads(r.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"  [phext write error] {e}")
        return False


def phext_read(coordinate: str) -> str:
    req = urllib.request.Request(
        f"{PHEXT_URL}/api/scroll/{coordinate}",
        headers={"authorization": PHEXT_TOKEN},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            return data.get("content", "")
    except Exception:
        return ""


# ── results.tsv parsing ─────────────────────────────────────────────────────────

def read_results(path="results.tsv") -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)
    return rows


def best_kept(results: list[dict]) -> dict | None:
    kept = [r for r in results if r.get("status") == "keep"]
    if not kept:
        return None
    return min(kept, key=lambda r: float(r.get("val_bpb", 999)))


def overnight_results(results: list[dict], since_hours: int = 12) -> list[dict]:
    """Results from the last N hours (approximate by recency — last N rows)."""
    cutoff = max(0, len(results) - since_hours * 12)  # ~12 experiments/hour
    return results[cutoff:]


# ── log tail ───────────────────────────────────────────────────────────────────

def tail_log(n=10) -> str:
    if not os.path.exists("run.log"):
        return "(no run.log)"
    result = subprocess.run(["tail", f"-n{n}", "run.log"],
                            capture_output=True, text=True)
    return result.stdout.strip()


def current_val_bpb() -> str:
    if not os.path.exists("run.log"):
        return "unknown"
    result = subprocess.run(["grep", "^val_bpb:", "run.log"],
                            capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    return lines[-1].replace("val_bpb:", "").strip() if lines else "in-progress"


# ── heartbeat ──────────────────────────────────────────────────────────────────

def run_heartbeat(node_id: int, collapse: bool = False):
    now = datetime.datetime.now()
    hour = now.hour
    label = HEARTBEAT_LABELS.get(hour, f"{hour}h")
    node_name = NODE_NAMES.get(node_id, f"Node {node_id}")
    pipe = NODE_PIPES.get(node_id, "?")

    results = read_results()
    best = best_kept(results)
    recent = overnight_results(results)
    recent_kept = [r for r in recent if r.get("status") == "keep"]
    recent_crashes = [r for r in recent if r.get("status") == "crash"]

    val_bpb_now = current_val_bpb()

    status_lines = [
        f"[{now.strftime('%Y-%m-%d %H:%M CT')}] {label.upper()} HEARTBEAT",
        f"Node:       {node_name} ({pipe})",
        f"Coord:      {node_log_coord(node_id)}",
        "",
        f"Current run:  val_bpb = {val_bpb_now}",
        f"Best ever:    val_bpb = {best['val_bpb'] if best else 'no results yet'}",
        f"  commit:     {best['commit'] if best else '-'}",
        f"  desc:       {best['description'] if best else '-'}",
        "",
        f"Recent (last ~12h):",
        f"  total:   {len(recent)} experiments",
        f"  kept:    {len(recent_kept)}",
        f"  crashes: {len(recent_crashes)}",
        "",
        f"Log tail:",
        tail_log(5),
    ]

    status = "\n".join(status_lines)
    print(status)

    # Write to heartbeat coordinate (morning/evening/midnight scroll within node's section)
    hb_coord = heartbeat_coord(node_id, hour)
    ok = phext_write(hb_coord, status)
    print(f"\n[phext] wrote to {hb_coord}: {'ok' if ok else 'FAILED'}")

    # Also write to node's overnight log coordinate
    ok2 = phext_write(node_log_coord(node_id), status)
    print(f"[phext] wrote to {node_log_coord(node_id)}: {'ok' if ok2 else 'FAILED'}")

    # Morning collapse
    if collapse and label == "morning":
        run_collapse(node_id, results)


def run_collapse(reporting_node: int, local_results: list[dict]):
    """
    Read all 9 S9RP node coordinates, find best val_bpb, write to collapse coord.
    """
    print("\n=== MORNING COLLAPSE ===")
    best_global = {"val_bpb": 999.0, "node": None, "text": ""}

    # Read own best
    own_best = best_kept(local_results)
    if own_best:
        bpb = float(own_best["val_bpb"])
        if bpb < best_global["val_bpb"]:
            best_global = {
                "val_bpb": bpb,
                "node": NODE_NAMES[reporting_node],
                "text": f"commit={own_best['commit']} | {own_best['description']}",
            }

    # Read other nodes from their result coords (9.1.1/1.1.N/1.1.9)
    for n in range(1, 10):
        if n == reporting_node:
            continue
        content = phext_read(node_result_coord(n))
        if not content:
            continue
        # Parse val_bpb line from status
        for line in content.splitlines():
            if "val_bpb =" in line.lower() and "best ever" in line.lower():
                try:
                    bpb = float(line.split("=")[-1].strip())
                    if bpb < best_global["val_bpb"]:
                        best_global = {
                            "val_bpb": bpb,
                            "node": NODE_NAMES.get(n, f"Node {n}"),
                            "text": line.strip(),
                        }
                except ValueError:
                    pass

    now = datetime.datetime.now()
    collapse_content = (
        f"[{now.strftime('%Y-%m-%d')}] MORNING COLLAPSE\n"
        f"Best val_bpb: {best_global['val_bpb']:.6f}\n"
        f"Node:         {best_global['node']}\n"
        f"Detail:       {best_global['text']}\n"
        f"\nAll nodes: pull new baseline from winning node before tonight's run."
    )
    print(collapse_content)
    ok = phext_write(COLLAPSE_COORD, collapse_content)
    print(f"\n[phext] wrote collapse to {COLLAPSE_COORD}: {'ok' if ok else 'FAILED'}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, default=1, help="Node ID (1-9, default: 1=Theia)")
    parser.add_argument("--collapse", action="store_true", help="Run morning collapse (8 AM only)")
    parser.add_argument("--check", action="store_true", help="Print status without writing phext")
    args = parser.parse_args()

    if args.check:
        results = read_results()
        best = best_kept(results)
        print(f"Node {args.node} ({NODE_NAMES.get(args.node,'?')})")
        print(f"Best val_bpb: {best['val_bpb'] if best else 'none'}")
        print(f"Current run:  {current_val_bpb()}")
        return

    run_heartbeat(args.node, collapse=args.collapse)


if __name__ == "__main__":
    main()
