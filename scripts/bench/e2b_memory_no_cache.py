"""E2b — tf.data memory footprint WITHOUT `.cache()`.

Ablation for E2: removes the single `.cache()` call from the tf.data pipeline
to isolate "TF runtime overhead" from "this project's caching choice." Same
methodology as E2, same iteration count, same sampling cadence.

Compare the resulting peak RSS against E2's Grain number to get the fair
loader-vs-loader comparison.

Run
---
    python scripts/bench/e2b_memory_no_cache.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

N_BATCHES = 50


def _read_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def _children(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            kids = [int(x) for x in f.read().split()]
    except FileNotFoundError:
        return []
    out = list(kids)
    for k in kids:
        out.extend(_children(k))
    return out


def _total_rss_mb(root_pid: int) -> tuple[float, int]:
    pids = [root_pid] + _children(root_pid)
    return sum(_read_rss_kb(p) for p in pids) / 1024.0, len(pids)


def main() -> int:
    from config import Config
    config = Config()
    grad_accum_steps = config.grad_accum_steps
    pid = os.getpid()

    print(f"[E2b] tf.data WITHOUT .cache()  pid={pid}")
    baseline, _ = _total_rss_mb(pid)
    print(f"[E2b] baseline (before imports)              : {baseline:7.1f} MB")

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    after_import, _ = _total_rss_mb(pid)
    print(f"[E2b] after import tensorflow                : {after_import:7.1f} MB   (+{after_import - baseline:.1f})")

    path = config.tfds_path.as_posix() + "_train"
    ds = (
        tf.data.Dataset.load(path)
        .rebatch(config.batch_size, drop_remainder=True)
        .batch(grad_accum_steps, drop_remainder=True)
        # NO .cache() here — that is the ablation
        .shuffle(10_000)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    it = iter(ds)
    after_build, _ = _total_rss_mb(pid)
    print(f"[E2b] after dataset+iterator build (no cache): {after_build:7.1f} MB   (+{after_build - after_import:.1f})")

    peak = after_build
    samples = []
    t0 = time.time()
    for i in range(N_BATCHES):
        b = next(it)
        # touch the tensors so they're actually materialized (matches E2's _NumpyIter behavior)
        _ = {k: v.numpy() for k, v in b.items()}
        if (i + 1) % 5 == 0:
            rss, _ = _total_rss_mb(pid)
            samples.append((i + 1, rss))
            if rss > peak:
                peak = rss
    t = time.time() - t0

    print()
    print(f"{'batch':>6} {'total RSS (MB)':>16}")
    for s, rss in samples:
        print(f"{s:>6} {rss:>16.1f}")
    print()
    print(f"[E2b] iterated {N_BATCHES} batches in {t:.2f}s")
    print(f"[E2b] PEAK total RSS                         : {peak:7.1f} MB")
    print(f"[E2b] Δ vs baseline (pure TF loader cost)    : {peak - baseline:+.1f} MB")
    print()
    print(f"Compare with E2 numbers:")
    print(f"  Grain peak (E2)            :   667 MB")
    print(f"  tf.data WITH .cache()  (E2):  16,303 MB")
    print(f"  tf.data WITHOUT .cache()   :  {peak:.0f} MB  ← this run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
