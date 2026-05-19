"""E2 — process memory footprint of the data pipeline.

Quantifies the "no TensorFlow dependency" claim by sampling VmRSS during a
50-batch iteration. The number you care about for capacity planning is
**peak RSS of the main process plus all child workers** — that is what shows
up against the pod / container memory budget.

Run
---
    LOADER_KIND=grain python scripts/bench/e2_memory_footprint.py
    LOADER_KIND=tf    python scripts/bench/e2_memory_footprint.py
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

LOADER_KIND = os.environ.get("LOADER_KIND", "grain").lower()
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
    # recurse one level (Grain spawns mp_prefetch workers under the main pid)
    all_kids = list(kids)
    for k in kids:
        all_kids.extend(_children(k))
    return all_kids


def _total_rss_mb(root_pid: int) -> tuple[float, int]:
    pids = [root_pid] + _children(root_pid)
    total = sum(_read_rss_kb(p) for p in pids) / 1024.0
    return total, len(pids)


def main() -> int:
    from config import Config
    config = Config()
    grad_accum_steps = config.grad_accum_steps
    pid = os.getpid()

    print(f"[E2] LOADER_KIND={LOADER_KIND}  pid={pid}")
    baseline, n0 = _total_rss_mb(pid)
    print(f"[E2] baseline (before imports)              : {baseline:7.1f} MB across {n0} pids")

    if LOADER_KIND == "grain":
        import grain
        from datasets import load_from_disk
    else:
        import data_tf  # imports tensorflow
    after_import, n1 = _total_rss_mb(pid)
    print(f"[E2] after import                           : {after_import:7.1f} MB across {n1} pids   (+{after_import - baseline:.1f})")

    if LOADER_KIND == "grain":
        hf_train = load_from_disk(config.dataset_path.as_posix() + "_train").with_format("numpy")
        ds = (
            grain.MapDataset.source(hf_train)
            .seed(config.seed)
            .shuffle()
            .repeat()
            .batch(config.batch_size * grad_accum_steps, drop_remainder=True)
            .to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=16))
            .mp_prefetch(grain.MultiprocessingOptions(num_workers=2))
        )
        it = iter(ds)
    else:
        it = data_tf.build_train_iter(config, grad_accum_steps)

    after_build, n2 = _total_rss_mb(pid)
    print(f"[E2] after dataset+iterator build           : {after_build:7.1f} MB across {n2} pids   (+{after_build - after_import:.1f})")

    # Iterate; sample RSS every 5 batches; track peak.
    peak = after_build
    peak_pids = n2
    samples = []
    t0 = time.time()
    for i in range(N_BATCHES):
        _ = next(it)
        if (i + 1) % 5 == 0:
            rss, nproc = _total_rss_mb(pid)
            samples.append((i + 1, rss, nproc))
            if rss > peak:
                peak, peak_pids = rss, nproc
    t = time.time() - t0

    print()
    print(f"{'batch':>6} {'total RSS (MB)':>16} {'pids':>5}")
    for s, rss, np_ in samples:
        print(f"{s:>6} {rss:>16.1f} {np_:>5}")
    print()
    print(f"[E2] iterated {N_BATCHES} batches in {t:.2f}s")
    print(f"[E2] PEAK total RSS                         : {peak:7.1f} MB across {peak_pids} pids")
    print(f"[E2] Δ vs baseline                          : {peak - baseline:+.1f} MB  (attributable to loader stack)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
