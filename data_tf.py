"""TF data pipeline — faithful port of the reference at
/mnt/project/kchat/edw/moe_nlp/matmoe_prod (data01.py + train_kl_k.py).

Reads TFDS shards produced by `prepare_tfds_from_arrow.py`. The train shards
are pre-batched at `config.compress_batch` (4096); the train pipeline rechunks
them with `.rebatch(B).batch(G)` so each yielded batch already has the
macro-step layout `[G, B, S]` — no `to_macro` reshape required (and the
existing `to_macro = arr.reshape(G, B, -1)` is a no-op identity for this
shape, so the train loop works unchanged across loaders).

Eval/test shards were saved at `eval_batch_size`; we `.rebatch` defensively in
case `config.eval_batch_size` changes between prep and training.

Selected with LOADER_KIND=tf at training time.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


def _load_train(config, grad_accum_steps: int) -> tf.data.Dataset:
    path = config.tfds_path.as_posix() + "_train"
    return (
        tf.data.Dataset.load(path)
        .rebatch(config.batch_size, drop_remainder=True)
        .batch(grad_accum_steps, drop_remainder=True)
        .cache()
        .shuffle(10_000)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )


def _load_eval(tfds_path: str, batch_size: int) -> tf.data.Dataset:
    return (
        tf.data.Dataset.load(tfds_path)
        .rebatch(batch_size, drop_remainder=False)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )


class _NumpyIter:
    """Wrap a tf.data iterator; yield numpy dicts. Stubs match Grain's API."""

    def __init__(self, ds: tf.data.Dataset):
        self._ds = ds
        self._it = iter(ds)

    def __iter__(self):
        return self

    def __next__(self):
        b = next(self._it)
        return {k: v.numpy() for k, v in b.items()}

    def get_state(self):
        return {}

    def set_state(self, _state):
        pass

    def start_prefetch(self):
        pass

    def close(self):
        pass


class _ReiterableNumpy:
    """Eval/test wrapper — each iter() produces a fresh numpy iterator."""

    def __init__(self, ds: tf.data.Dataset):
        self._ds = ds

    def __iter__(self):
        for b in self._ds:
            yield {k: v.numpy() for k, v in b.items()}


def build_train_iter(config, grad_accum_steps: int) -> _NumpyIter:
    return _NumpyIter(_load_train(config, grad_accum_steps))


def build_eval_iter(tfds_path: str, batch_size: int) -> _ReiterableNumpy:
    return _ReiterableNumpy(_load_eval(tfds_path, batch_size))
