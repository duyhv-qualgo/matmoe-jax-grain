"""E1b — multi-cycle resume stress test.

Extends E1 from a single restore to repeated preempt/resume cycles, the real
production pattern on shared/preemptible infra. We run a 200-batch reference
stream end-to-end, then re-do it as four contiguous 50-batch segments where
each segment restores from the previous segment's saved state. Every batch
across all four segments must match the reference bit-for-bit.

This proves the resume mechanism is repeatable, not a one-off coincidence,
and that saved state stays valid across many cycles (no state-object drift,
no shared-memory leak compounding, etc.).

Run
---
    LOADER_KIND=grain python scripts/bench/e1b_resume_stress.py
    LOADER_KIND=tf    python scripts/bench/e1b_resume_stress.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config import Config  # noqa: E402

N_TOTAL = 200
CYCLE = 50          # restore every CYCLE batches
LOADER_KIND = os.environ.get("LOADER_KIND", "grain").lower()
STATE_DIR = ROOT / "tmp" / f"e1b_{LOADER_KIND}"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_batch(batch: dict) -> str:
    h = hashlib.sha1()
    for k in sorted(batch.keys()):
        v = np.asarray(batch[k])
        h.update(k.encode())
        h.update(str(v.shape).encode())
        h.update(str(v.dtype).encode())
        h.update(v.tobytes())
    return h.hexdigest()[:16]


def build_grain_iter(config: Config, grad_accum_steps: int):
    import grain
    from datasets import load_from_disk
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
    return iter(ds)


def build_tf_iter(config: Config, grad_accum_steps: int):
    import data_tf
    return data_tf.build_train_iter(config, grad_accum_steps)


def main() -> int:
    config = Config()
    grad_accum_steps = config.grad_accum_steps
    builder = build_grain_iter if LOADER_KIND == "grain" else build_tf_iter

    n_cycles = N_TOTAL // CYCLE
    print(f"[E1b] LOADER_KIND={LOADER_KIND}  N_TOTAL={N_TOTAL}  CYCLE={CYCLE}  cycles={n_cycles}")

    # --- Reference: single uninterrupted stream of N_TOTAL batches.
    t0 = time.time()
    it_ref = builder(config, grad_accum_steps)
    ref_hashes = []
    save_points = list(range(CYCLE, N_TOTAL, CYCLE))  # save after every CYCLE except the last
    for i in range(N_TOTAL):
        b = next(it_ref)
        ref_hashes.append(_hash_batch(b))
        if (i + 1) in save_points:
            state = it_ref.get_state()
            p = STATE_DIR / f"state_after_{i + 1}.msgpack"
            with open(p, "wb") as f:
                f.write(msgpack.packb(state, use_bin_type=True))
    t_ref = time.time() - t0
    print(f"[E1b] reference stream: {N_TOTAL} batches in {t_ref:.2f}s")

    # --- Replay: rebuild iterator and restore from each save point.
    # Cycle 0 is fresh start (batches 1..50), then for each save_point at K,
    # build new iterator, restore, take CYCLE batches (K+1..K+CYCLE).
    replay_hashes = []

    # Cycle 0 — fresh iterator, no restore.
    it0 = builder(config, grad_accum_steps)
    for _ in range(CYCLE):
        replay_hashes.append(_hash_batch(next(it0)))
    del it0

    for sp in save_points:
        with open(STATE_DIR / f"state_after_{sp}.msgpack", "rb") as f:
            state = msgpack.unpackb(f.read(), raw=False)
        it = builder(config, grad_accum_steps)
        it.set_state(state)
        for _ in range(CYCLE):
            replay_hashes.append(_hash_batch(next(it)))
        del it

    # --- Diff.
    assert len(replay_hashes) == len(ref_hashes) == N_TOTAL
    matches = [a == b for a, b in zip(ref_hashes, replay_hashes)]

    per_cycle = []
    for c in range(n_cycles):
        lo, hi = c * CYCLE, (c + 1) * CYCLE
        per_cycle.append(sum(matches[lo:hi]))

    print()
    print(f"{'cycle':>6} {'range':>13} {'matched':>9}")
    for c, m in enumerate(per_cycle):
        lo, hi = c * CYCLE + 1, (c + 1) * CYCLE
        tag = "fresh start" if c == 0 else f"restore@{c * CYCLE}"
        print(f"{c:>6} {lo:>4}..{hi:<4}  {m:>3}/{CYCLE}   ({tag})")

    total_match = sum(matches)
    print()
    print(f"[E1b][{LOADER_KIND}] total {total_match}/{N_TOTAL} batches match reference")

    if LOADER_KIND == "grain":
        ok = total_match == N_TOTAL
        print(f"[E1b][grain] {'PASS' if ok else 'FAIL'} — {'repeated resume stays bit-identical' if ok else 'resume mechanism drifted'}")
        return 0 if ok else 1

    if total_match == CYCLE:
        # Cycle 0 (fresh) matches; later cycles diverge — exactly the documented gap.
        print("[E1b][tf] EXPECTED — only the fresh-start cycle matches; tf.data restore is a no-op")
    elif total_match == N_TOTAL:
        print("[E1b][tf] UNEXPECTED — tf wrapper appears to have resumed")
    else:
        print("[E1b][tf] partial overlap (likely deterministic shuffle on rebuild) — restore not real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
