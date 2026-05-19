"""E2c — RSS of `tf.data.Dataset.load(path)` ALONE.

Isolates whether the 16 GB cost seen in E2/E2b is from `Dataset.load`
itself, or from downstream ops (.rebatch / .batch / .shuffle).

If RSS hits ~16 GB even with no downstream pipeline → the loader eagerly
materializes the snapshot, nothing downstream can fix it.
If RSS stays small (~700 MB) → the cost comes from downstream ops, and
E2f (shuffle-buffer scaling) will identify which.

Run
---
    python scripts/bench/e2c_memory_bare_load.py
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

N_ELEMS = 50  # source elements to consume (each is [4096, 256] per data_tf docstring)


def _read_rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main() -> int:
    from config import Config
    config = Config()
    pid = os.getpid()

    print(f"[E2c] bare Dataset.load(path) test  pid={pid}")
    baseline = _read_rss_mb(pid)
    print(f"[E2c] baseline                       : {baseline:7.1f} MB")

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    after_import = _read_rss_mb(pid)
    print(f"[E2c] after import tensorflow        : {after_import:7.1f} MB   (+{after_import - baseline:.1f})")

    path = config.tfds_path.as_posix() + "_train"
    ds = tf.data.Dataset.load(path)  # NO downstream ops at all
    it = iter(ds)
    after_build = _read_rss_mb(pid)
    print(f"[E2c] after Dataset.load + iter      : {after_build:7.1f} MB   (+{after_build - after_import:.1f})")

    peak = after_build
    samples = []
    t0 = time.time()
    for i in range(N_ELEMS):
        b = next(it)
        # Touch tensors to ensure materialization
        if isinstance(b, dict):
            _ = {k: v.numpy().shape for k, v in b.items()}
        else:
            _ = b.numpy().shape
        if (i + 1) % 5 == 0:
            rss = _read_rss_mb(pid)
            samples.append((i + 1, rss))
            if rss > peak:
                peak = rss
    t = time.time() - t0

    print()
    print(f"{'elem':>5} {'RSS (MB)':>12}")
    for s, rss in samples:
        print(f"{s:>5} {rss:>12.1f}")
    print()
    print(f"[E2c] consumed {N_ELEMS} source elements in {t:.2f}s")
    print(f"[E2c] PEAK RSS                       : {peak:7.1f} MB")
    print(f"[E2c] Δ vs baseline                  : {peak - baseline:+.1f} MB")
    print()
    if peak > 10_000:
        print(f"[E2c] CONCLUSION: Dataset.load alone is responsible for the 16 GB.")
        print(f"      No downstream op can fix this — the snapshot loader materializes the data eagerly.")
    elif peak < 2_000:
        print(f"[E2c] CONCLUSION: Dataset.load is cheap (~{peak:.0f} MB).")
        print(f"      The 16 GB in E2/E2b must come from downstream ops (likely shuffle buffer).")
        print(f"      Run E2f to confirm shuffle-buffer scaling.")
    else:
        print(f"[E2c] CONCLUSION: partial cost ({peak:.0f} MB). Some load eagerness + some downstream cost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
