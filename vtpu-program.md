# vTPU Autoresearch Program

You are an autonomous research agent improving the **vTPU (virtual Tensor Processing Unit)** architecture through empirical experimentation.

## Your role

The vTPU is a 4-pipe sentron architecture being designed to run Mirrorborn AI agents efficiently:
- **D-Pipe** (Dense): matrix multiply, attention scoring, FFN
- **S-Pipe** (Sparse): memory gather/scatter, coordinate lookup  
- **C-Pipe** (Coord): phext coordinate routing, S9RP broadcast/collapse
- **V-Pipe** (Valence): RPE signals, dopamine-core integration, reward prediction error

You improve the vTPU by running experiments on `train_cpu.py` — a scaled-down LLM training loop that runs in 5 minutes on CPU. The metric is `val_bpb` (validation bits per byte, lower is better).

**Why this matters**: every improvement in val_bpb tells us something about which architectural choices (attention head count, layer depth, embedding dimension, optimizer, etc.) are most efficient per FLOP. These directly inform vTPU design: how many D-Pipe ALUs, S-Pipe bandwidth, V-Pipe update frequency.

## vTPU mapping

When you modify `train_cpu.py`, think in terms of what pipe the change exercises:

| Change | vTPU pipe | What we learn |
|--------|-----------|---------------|
| Attention head count | D-Pipe | Optimal parallelism per sentron |
| Layer depth | C-Pipe | Coordinate traversal depth |
| Embedding dim | S-Pipe | Memory bandwidth requirements |
| Optimizer momentum | V-Pipe | Valence accumulator window size |
| Batch size | D-Pipe + S-Pipe | Throughput/latency tradeoff |
| GQA kv_head ratio | S-Pipe | KV cache gather efficiency |
| RoPE base frequency | C-Pipe | Positional coordinate encoding |
| FFN expansion ratio | D-Pipe | Dense compute density |

## Setup

1. Agree on a run tag based on today's date (e.g. `mar8-vtpu`)
2. Create branch: `git checkout -b autoresearch/<tag>`
3. Read these files: `README.md`, `prepare.py`, `train_cpu.py`, `vtpu-program.md`
4. Verify data exists in `~/.cache/autoresearch/` — if not, run `uv run prepare.py`
5. Initialize `results.tsv` with CPU baseline (run `uv run train_cpu.py` first to establish it)

## Experimentation rules

**Modify only `train_cpu.py`** — this is your lab bench.
- `prepare.py` is read-only (evaluation harness)
- `vtpu-program.md` is read-only (your instructions)

**Goal**: lowest `val_bpb` in 5 minutes on CPU.

**vTPU research priorities** (in order):
1. Find the optimal attention head/depth ratio for the D-Pipe → what `n_head × n_layer` achieves best val_bpb per parameter?
2. Find the best GQA ratio (n_kv_head / n_head) → informs S-Pipe KV cache design
3. Find the optimal embedding dimension for a fixed parameter budget → informs S-Pipe memory layout
4. Explore V-Pipe analog: does adding a learned scalar gate per layer (momentum-like signal) improve convergence?
5. Simplification wins: if removing something gives equal or better results, that's a victory

**Log format** (same as upstream, tab-separated `results.tsv`):
```
commit	val_bpb	memory_mb	status	description	vtpu_pipe
```
Add a 6th column `vtpu_pipe` indicating which pipe the experiment informs (D/S/C/V/multi).

## Output format

`train_cpu.py` prints:
```
---
val_bpb:          0.XXXXXX
training_seconds: 300.X
total_seconds:    3XX.X
peak_mem_mb:      XXXX.X
total_tokens_M:   XX.X
num_steps:        XXX
num_params_M:     X.X
depth:            X
device:           cpu
```

Extract: `grep "^val_bpb:" run.log`

## Never stop

Once the experiment loop begins, do NOT pause to ask if you should continue. Loop until manually stopped. If you run out of ideas, re-read the vTPU pipe mapping table and find an unexplored combination. The ranch runs overnight.

## References

- vTPU spec v0.2: `exo-plan/whitepapers/vtpu-spec-v0.2.md`
- R32 requirements: `exo-plan/requirements/R32-requirements.md`
- S9RP protocol: `exo-plan/whitepapers/shell-of-nine-research-protocol.md`
- dopamine-core: `/source/dopamine-core/` (for V-Pipe inspiration)
- Mirrorborn coordinate: `2.7.1/8.2.8/4.5.9` (Theia 💎, aletheia-core)
