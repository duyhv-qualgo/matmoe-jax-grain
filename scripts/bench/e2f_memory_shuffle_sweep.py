"""E2f — RSS as a function of shuffle buffer size.

If shuffle is the dominant cost, RSS should scale roughly linearly with
buffer size. If shuffle is irrelevant (Dataset.load eagerness is the
cause), RSS should be flat across buffer sizes.

Buffer sizes tested: 0 (no shuffle), 100, 1_000, 10_000 (production).

Run
---
    python scripts/bench/e2f_memory_shuffle_sweep.py
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

N_BATCHES = 30  # smaller — we run this 4 times


def _read_rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def _run_one(tf, config, grad_accum_steps: int, shuffle_buf: int, pid: int) -> tuple[float, float]:
    """Build pipeline with given shuffle buffer, iterate, return (after_build_MB, peak_MB)."""
    path = config.tfds_path.as_posix() + "_train"
    ds = (
        tf.data.Dataset.load(path)
        .rebatch(config.batch_size, drop_remainder=True)
        .batch(grad_accum_steps, drop_remainder=True)
    )
    if shuffle_buf > 0:
        ds = ds.shuffle(shuffle_buf)
    ds = ds.repeat().prefetch(tf.data.AUTOTUNE)
    it = iter(ds)

    after_build = _read_rss_mb(pid)
    peak = after_build
    for _ in range(N_BATCHES):
        b = next(it)
        _ = {k: v.numpy().shape for k, v in b.items()}
        rss = _read_rss_mb(pid)
        if rss > peak:
            peak = rss
    # Release the iterator before returning so the next run starts cleaner.
    del it, ds
    return after_build, peak


def main() -> int:
    from config import Config
    config = Config()
    grad_accum_steps = config.grad_accum_steps
    pid = os.getpid()

    print(f"[E2f] shuffle-buffer RSS sweep  pid={pid}")
    baseline = _read_rss_mb(pid)
    print(f"[E2f] baseline                       : {baseline:7.1f} MB")

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    after_import = _read_rss_mb(pid)
    print(f"[E2f] after import tensorflow        : {after_import:7.1f} MB   (+{after_import - baseline:.1f})")
    print()

    results = []
    for buf in [0, 100, 1_000, 10_000]:
        t0 = time.time()
        ab, peak = _run_one(tf, config, grad_accum_steps, buf, pid)
        t = time.time() - t0
        results.append((buf, ab, peak))
        label = "(no shuffle)" if buf == 0 else f"shuffle({buf:,})"
        print(f"[E2f] {label:<18} after_build={ab:8.1f} MB   peak={peak:8.1f} MB   ({t:.1f}s)")

    print()
    print(f"{'buffer':>10} {'after_build':>14} {'peak':>10} {'Δ vs no-shuffle':>18}")
    base_peak = results[0][2]
    for buf, ab, peak in results:
        delta = peak - base_peak
        label = "0" if buf == 0 else f"{buf:,}"
        print(f"{label:>10} {ab:>14.1f} {peak:>10.1f} {delta:>+18.1f}")

    print()
    growth = results[-1][2] - results[0][2]
    print(f"[E2f] Δ RSS from shuffle(0) to shuffle(10_000) = {growth:+.1f} MB")
    if growth > 1_000:
        print(f"      → shuffle buffer accounts for ~{growth:.0f} MB of the gap (significant)")
    if results[0][2] > 10_000:
        print(f"      → even with NO shuffle, RSS is {results[0][2]:.0f} MB → bulk of cost is upstream of shuffle (Dataset.load/rebatch/batch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
