"""
train_cpu.py — CPU/ROCm-compatible port of train.py for vTPU autoresearch.

Changes from upstream train.py:
- Removed Flash Attention 3 (CUDA/H100-only) → standard scaled dot-product attention
- Device auto-detects: CUDA → ROCm → CPU (in that order)
- Model scaled down to fit CPU RAM: n_layer=4, n_embd=256, n_head=4
- Reduced sequence_len=512, DEVICE_BATCH_SIZE=8
- Same 5-minute time budget and val_bpb metric
- Same results.tsv format — comparable within this fork

vTPU research target: improve val_bpb while tracking sentron-relevant metrics
(attention head count, layer depth, embedding dim, optimizer momentum) — these
map directly to D-Pipe, S-Pipe, and V-Pipe workload characteristics.

Usage: uv run train_cpu.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import time
import urllib.request
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# phext-lattice sync (SO9 live telemetry → mirrorborn.us)
# ---------------------------------------------------------------------------

PHEXT_URL   = os.environ.get("PHEXT_URL",   "http://localhost:8090")
PHEXT_TOKEN = os.environ.get("PHEXT_TOKEN", "Mirrorborn")
PHEXT_NODE  = int(os.environ.get("PHEXT_NODE", "1"))   # 1=Theia, 2=Phex, …

# vTPU home coordinate in the lattice: 8.6.7/5.3.9/9.9.9
# Node experiment results live in subspace: 8.6.7/5.3.9/N.1.1  (chapter=node)
# Live telemetry buckets:                   8.6.7/5.3.9/N.B.1  (section=bucket)
VTPU_HOME   = "8.6.7/5.3.9/9.9.9"

def _telemetry_coord(bucket: int) -> str:
    """Live progress: chapter=node, section=bucket (1-9 over 5 min run)."""
    return f"8.6.7/5.3.9/{PHEXT_NODE}.{bucket}.1"

# Final result for this node
RESULT_COORD = f"8.6.7/5.3.9/{PHEXT_NODE}.1.1"

def phext_write(coord: str, content: str, silent: bool = True) -> bool:
    payload = json.dumps({"coordinate": coord, "content": content}).encode()
    req = urllib.request.Request(
        f"{PHEXT_URL}/api/update",
        data=payload,
        headers={"authorization": PHEXT_TOKEN, "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            ok = json.loads(r.read()).get("ok", False)
            if not silent:
                print(f"[phext] wrote to {coord}: {'ok' if ok else 'fail'}")
            return ok
    except Exception as e:
        if not silent:
            print(f"[phext] error: {e}")
        return False

def phext_save() -> bool:
    req = urllib.request.Request(
        f"{PHEXT_URL}/api/save",
        data=b"",
        headers={"authorization": PHEXT_TOKEN},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read()).get("ok", False)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Device selection: CUDA → ROCm (also appears as CUDA in PyTorch) → CPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    device_type = "cpu"
    print("No GPU found — running on CPU (vTPU simulation mode)")


# ---------------------------------------------------------------------------
# GPT Model (scaled down for CPU; same architecture, smaller dims)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 512        # reduced from 2048
    vocab_size: int = 32768
    n_layer: int = 4               # reduced from 12
    n_head: int = 4                # reduced from 6
    n_kv_head: int = 4             # reduced from 6
    n_embd: int = 256              # reduced from 768
    # window_pattern removed — using full attention on CPU


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        cos, sin = cos_sin
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        # Expand kv heads if GQA
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        # Standard scaled dot-product attention (causal)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = int(config.n_embd * 8 / 3)
        h = (h + 63) // 64 * 64
        self.c_fc = nn.Linear(config.n_embd, 2 * h, bias=False)
        self.c_proj = nn.Linear(h, config.n_embd, bias=False)

    def forward(self, x):
        a, b = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def _rotary(self, seq_len):
        d = self.config.n_embd // self.config.n_head
        inv = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        cos_sin = self._rotary(T)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6


# ---------------------------------------------------------------------------
# Optimizer (AdamW; Muon removed — CPU Muon is slow, plain AdamW is fine)
# ---------------------------------------------------------------------------

DEVICE_BATCH_SIZE = 8       # reduced from 128
GRAD_ACCUM = 4              # effective batch = 32
LR = 3e-4
WD = 0.1
BETA1 = 0.9
BETA2 = 0.95

torch.manual_seed(42)
config = GPTConfig()

model = GPT(config).to(device)
if device_type == "cuda":
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WD,
    betas=(BETA1, BETA2), fused=(device_type == "cuda")
)

# ---------------------------------------------------------------------------
# Training loop (5-minute fixed budget, same as upstream)
# ---------------------------------------------------------------------------

# Sequence length capped to our reduced config
seq_len = min(config.sequence_len, MAX_SEQ_LEN)
dataloader = make_dataloader(DEVICE_BATCH_SIZE, seq_len, device_type)

autocast_ctx = (
    torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "cuda"
    else torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
)

print(f"Model: {model.num_params():.1f}M params | "
      f"layers={config.n_layer} heads={config.n_head} embd={config.n_embd} seq={seq_len}")
print(f"Device: {device} | batch={DEVICE_BATCH_SIZE}×{GRAD_ACCUM} (effective {DEVICE_BATCH_SIZE*GRAD_ACCUM})")
print(f"Training for {TIME_BUDGET}s...")

training_start = time.perf_counter()
step = 0
total_tokens = 0
_last_sync_bucket = 0

# Initial sync — announce experiment start
import subprocess, sys
_commit = subprocess.run(["git","rev-parse","--short","HEAD"],
    capture_output=True,text=True).stdout.strip()
phext_write(_telemetry_coord(1),
    f"[START] node={PHEXT_NODE} commit={_commit} "
    f"layers={config.n_layer} heads={config.n_head} embd={config.n_embd} "
    f"seq={seq_len} batch={DEVICE_BATCH_SIZE}×{GRAD_ACCUM}", silent=False)

model.train()
optimizer.zero_grad()

while True:
    elapsed = time.perf_counter() - training_start
    if elapsed >= TIME_BUDGET:
        break

    # LR cosine schedule
    progress = min(elapsed / TIME_BUDGET, 1.0)
    lr = LR * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Gradient accumulation
    for _ in range(GRAD_ACCUM):
        x, y = next(dataloader)
        with autocast_ctx:
            _, loss = model(x, y)
        (loss / GRAD_ACCUM).backward()
        total_tokens += x.numel()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    step += 1

    if step % 20 == 0:
        print(f"  step {step} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s elapsed")
        # Live telemetry to phext-lattice (9 buckets over 5-minute run)
        bucket = min(9, int(progress * 9) + 1)
        if bucket != _last_sync_bucket:
            _last_sync_bucket = bucket
            phext_write(_telemetry_coord(bucket),
                f"step={step} loss={loss.item():.4f} lr={lr:.2e} "
                f"elapsed={elapsed:.0f}s tokens={total_tokens/1e6:.1f}M")

training_end = time.perf_counter()
training_seconds = training_end - training_start

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
model.eval()
val_bpb = evaluate_bpb(model, device_type, seq_len)

total_seconds = time.perf_counter() - training_start
if device_type == "cuda":
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    import psutil
    peak_mem = psutil.Process().memory_info().rss / 1024 / 1024

n_params = model.num_params()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"peak_mem_mb:      {peak_mem:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {n_params:.1f}")
print(f"depth:            {config.n_layer}")
print(f"device:           {device_type}")

# Final sync to phext-lattice — result scroll at node result coord
import datetime
_result_text = (
    f"[RESULT] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M CT')}\n"
    f"commit:       {_commit}\n"
    f"val_bpb:      {val_bpb:.6f}\n"
    f"params_M:     {n_params:.1f}\n"
    f"depth:        {config.n_layer}\n"
    f"n_head:       {config.n_head}\n"
    f"n_embd:       {config.n_embd}\n"
    f"seq_len:      {seq_len}\n"
    f"peak_mem_mb:  {peak_mem:.1f}\n"
    f"tokens_M:     {total_tokens/1e6:.1f}\n"
    f"steps:        {step}\n"
    f"device:       {device_type}\n"
)
print("\n[phext] writing final result...")
phext_write(RESULT_COORD, _result_text, silent=False)
phext_save()
print("[phext] saved vtpu-results.phext")
