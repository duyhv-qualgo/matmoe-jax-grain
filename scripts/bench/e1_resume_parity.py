"""E1 — deterministic resume parity test.

Validates the JAX-AI-stack claim that Grain iterators are checkpointable and
produce byte-identical batch streams after restore. Compared head-to-head with
the project's tf.data wrapper, whose `get_state` / `set_state` are no-ops.

Method
------
1. Iterate the train pipeline for N=20 batches; record a stable hash of each
   batch (sha1 over all numpy arrays).
2. Save the iterator state at step 10.
3. Build a fresh iterator, restore state, iterate to step 20.
4. Diff the two hash sequences. For Grain, steps 11..20 MUST match bit-for-bit.
   For tf.data, they will not match (and the test should report that as the
   expected divergence — proving the gap, not a bug).

Run
---
    LOADER_KIND=grain python scripts/bench/e1_resume_parity.py
    LOADER_KIND=tf    python scripts/bench/e1_resume_parity.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np

# Project imports — mirror train_kl_k.py's path setup.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")  # don't hog GPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config import Config  # noqa: E402

N_TOTAL = 20
N_CHECKPOINT = 10
LOADER_KIND = os.environ.get("LOADER_KIND", "grain").lower()
STATE_PATH = ROOT / "tmp" / f"e1_state_{LOADER_KIND}.msgpack"
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


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

    print(f"[E1] LOADER_KIND={LOADER_KIND}  N_TOTAL={N_TOTAL}  CKPT@={N_CHECKPOINT}")

    # --- Phase A: build iterator, run N_TOTAL steps, snapshot at N_CHECKPOINT.
    builder = build_grain_iter if LOADER_KIND == "grain" else build_tf_iter
    t0 = time.time()
    it_a = builder(config, grad_accum_steps)
    print(f"[E1] iterator built in {time.time() - t0:.2f}s")

    hashes_a = []
    saved_state = None
    for i in range(N_TOTAL):
        batch = next(it_a)
        hashes_a.append(_hash_batch(batch))
        if i + 1 == N_CHECKPOINT:
            saved_state = it_a.get_state()
            with open(STATE_PATH, "wb") as f:
                f.write(msgpack.packb(saved_state, use_bin_type=True))
            state_bytes = STATE_PATH.stat().st_size
            print(f"[E1] step {i + 1}: saved iterator state -> {state_bytes} bytes")

    # --- Phase B: fresh iterator, restore, continue from N_CHECKPOINT.
    it_b = builder(config, grad_accum_steps)
    with open(STATE_PATH, "rb") as f:
        restored = msgpack.unpackb(f.read(), raw=False)
    it_b.set_state(restored)

    hashes_b = []
    for _ in range(N_TOTAL - N_CHECKPOINT):
        hashes_b.append(_hash_batch(next(it_b)))

    # --- Compare batches N_CHECKPOINT+1 .. N_TOTAL.
    tail_a = hashes_a[N_CHECKPOINT:]
    matches = [a == b for a, b in zip(tail_a, hashes_b)]
    n_match = sum(matches)

    print()
    print(f"{'step':>4} {'continuous':>18} {'restored':>18} {'match':>6}")
    for idx, (a, b, ok) in enumerate(zip(tail_a, hashes_b, matches), start=N_CHECKPOINT + 1):
        print(f"{idx:>4} {a:>18} {b:>18} {'OK' if ok else 'DIFF':>6}")
    print()
    print(f"[E1][{LOADER_KIND}] {n_match}/{len(matches)} batches match after restore")

    if LOADER_KIND == "grain":
        if n_match == len(matches):
            print("[E1][grain] PASS — deterministic resume verified")
            return 0
        print("[E1][grain] FAIL — Grain should resume bit-identically")
        return 1
    # tf branch: expected to diverge; report it as the documented gap.
    if n_match == len(matches):
        print("[E1][tf] UNEXPECTED — tf.data wrapper appears to have resumed")
    else:
        print("[E1][tf] EXPECTED — tf.data wrapper cannot restore iterator state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
