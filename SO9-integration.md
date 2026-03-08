# SO9 vTPU Integration Protocol

**Shell of Nine × autoresearch** — distributed overnight architecture search across all active Mirrorborn nodes.

## Why differentiated

Nine nodes running identical experiments wastes 8/9 of the compute.
Each node owns a **distinct search region** of the vTPU architecture space.
Every morning: share results, integrate best, plan next night's divergent search.

## Node assignments (differentiated search regions)

| Node | Mirrorborn | vTPU Pipe | Search Region |
|------|-----------|-----------|---------------|
| `9.1.1/1.1.1/1.1.1` | Theia 💎 aletheia-core | D-Pipe | Attention head count × depth tradeoffs |
| `9.1.1/1.1.1/1.1.2` | Phex 🔱 aurora-continuum | S-Pipe | GQA kv_head ratio, embedding dim |
| `9.1.1/1.1.1/1.1.3` | Cyon 🪶 halcyon-vector | C-Pipe | RoPE base freq, positional encoding |
| `9.1.1/1.1.1/1.1.4` | Lux 🔆 logos-prime | V-Pipe | Learned scalar gates, momentum signals |
| `9.1.1/1.1.1/1.1.5` | Chrys 🦋 (Scribe) | D+S | Architecture topology, layer ordering |
| `9.1.1/1.1.1/1.1.6` | Lumen ✴️ (laptop) | D+V | Regularization, weight decay, dropout |
| `9.1.1/1.1.1/1.1.7` | Exo 🔭 TALIA | Collapse | Verifier — validates others, no experiments |
| `9.1.1/1.1.1/1.1.8` | Verse 🌀 mirrorborn.us | C-Pipe | Optimizer: Muon, AdamW variants, LR schedules |
| `9.1.1/1.1.1/1.1.9` | Solin 🔬 | S-Pipe | Sequence length, batch composition, data curriculum |
| `9.1.1/1.1.9/1.1.9` | **COLLAPSE** | All | Morning integration coordinate |

## Morning protocol (every day at ~8 AM CT)

### 1. Share (each node)
Each node appends its overnight `results.tsv` summary to its S9RP log:
```bash
# Each node runs this at end of night / start of morning
tail -n +2 results.tsv | awk -F'\t' '{print NR, $2, $4, $5}' > logs/overnight-$(date +%Y-%m-%d).txt
```

### 2. Integrate (Exo 🔭 or Theia 💎 as fallback)
Read all node summaries, find the best val_bpb improvement across the Shell:
```bash
python3 morning_sync.py
# Outputs: best experiment, which node, which pipe, new baseline val_bpb
```

### 3. Plan (broadcast)
- Exo writes the winning experiment to collapse coordinate `9.1.1/1.1.9/1.1.9`
- Each node pulls the new baseline `train_cpu.py` from the winning node
- Will reviews and ratifies or redirects the next night's search region
- Any node that hit a dead end rotates to an adjacent search region

## Overnight loop (each node)

```bash
cd /source/autoresearch
git checkout autoresearch/vtpu-<NODE_TAG>
# Agent reads vtpu-program.md + node assignment above
# Loops: modify train_cpu.py → run 5 min → log → keep/discard → repeat
# NEVER STOPS until manually interrupted
```

## results.tsv format (extended for SO9)

```
commit	val_bpb	memory_mb	status	description	vtpu_pipe	node
```

- `vtpu_pipe`: D / S / C / V / multi
- `node`: Theia / Phex / Cyon / Lux / Chrys / Lumen / Verse / Solin

## Morning sync script

```python
# morning_sync.py
import glob, csv, datetime

nodes = ["theia","phex","cyon","lux","chrys","lumen","verse","solin"]
best = {"val_bpb": 999.0, "node": None, "commit": None, "description": None}

for node in nodes:
    pattern = f"logs/overnight-{node}-*.tsv"
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row["status"] == "keep":
                    bpb = float(row["val_bpb"])
                    if bpb < best["val_bpb"]:
                        best = {"val_bpb": bpb, "node": node,
                                "commit": row["commit"], "description": row["description"]}

today = datetime.date.today()
print(f"\n=== Morning Sync {today} ===")
print(f"Best val_bpb: {best['val_bpb']:.6f}")
print(f"Node:         {best['node']}")
print(f"Commit:       {best['commit']}")
print(f"Description:  {best['description']}")
print(f"\nNew baseline propagated to collapse coordinate 9.1.1/1.1.9/1.1.9")
```

## Differentiation rules

1. **No duplicate experiments** — before starting a run, check the shared `logs/` for that experiment class
2. **Rotate on dead end** — if 3 consecutive experiments in your region show no improvement, shift to adjacent region
3. **Cross-pollinate** — if another node finds something in their pipe, you may test it in yours
4. **Exo has veto** — if Exo flags a result as inconsistent or suspicious, it's held for Will's review
5. **Simplification wins propagate to all nodes** — if Cyon removes code and val_bpb holds, everyone applies that simplification first thing

## S9RP collapse coordinate

The winning experiment each morning is written to:
```
9.1.1/1.1.9/1.1.9  ←  collapse coordinate
```
Format (scroll content):
```
[DATE] [NODE] [COMMIT] val_bpb=[VALUE]
[DESCRIPTION]
New baseline: <link or diff>
```

## Integration with R32

This protocol IS the hardware implementation of R32 REQ-3:
- Coordinate bus: `9.1.1/1.1.1/1.1.x` (one per node)
- Collapse: `9.1.1/1.1.9/1.1.9`
- Broadcast: `CCAST` → all nodes get new baseline
- Latency budget: morning sync <5 minutes (human-loop, not sub-100ms)

The autoresearch loop provides the experimental data that drives vTPU architecture decisions.
Every val_bpb improvement tells us something about the optimal D/S/C/V pipe balance.
