"""E3 — time-to-first-batch comparison.

Measures cold-start latency for each loader. Useful for the report because it
quantifies the "no TensorFlow dependency" claim in a way managers can see:
fewer imports, no TF graph build, no TF session warmup.

Reports four checkpoints:
  T_import   — time to import the loader's library (grain vs tensorflow)
  T_build    — time to construct the dataset / iterator object
  T_first    — time from build complete to first batch yielded
  T_total    — sum of the above (process startup to first usable batch)

Run
---
    LOADER_KIND=grain python scripts/bench/e3_startup_time.py
    LOADER_KIND=tf    python scripts/bench/e3_startup_time.py
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


def main() -> int:
    from config import Config
    config = Config()
    grad_accum_steps = config.grad_accum_steps

    print(f"[E3] LOADER_KIND={LOADER_KIND}")

    t0 = time.time()
    if LOADER_KIND == "grain":
        import grain  # noqa: F401
        from datasets import load_from_disk
    else:
        import data_tf  # noqa: F401  (imports tensorflow internally)
    t_import = time.time() - t0

    t0 = time.time()
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
    t_build = time.time() - t0

    t0 = time.time()
    batch = next(it)
    t_first = time.time() - t0

    total = t_import + t_build + t_first
    print()
    print(f"  T_import : {t_import:7.3f} s   (import {'grain+datasets' if LOADER_KIND == 'grain' else 'data_tf(tensorflow)'})")
    print(f"  T_build  : {t_build:7.3f} s   (construct dataset + iterator)")
    print(f"  T_first  : {t_first:7.3f} s   (first batch yielded)")
    print(f"  T_total  : {total:7.3f} s   (cold start -> first usable batch)")
    print()
    sample_key = next(iter(batch))
    print(f"  Sanity   : got batch with keys {list(batch.keys())}; '{sample_key}' shape={batch[sample_key].shape} dtype={batch[sample_key].dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
