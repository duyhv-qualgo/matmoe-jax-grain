# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

JAX/Flax (NNX) Mixture-of-Experts encoder-decoder Transformer for English↔Vietnamese translation, trained on `ura-hcmut/PhoMT`. The "MatMoE" design combines a Matryoshka elastic MLP (multiple nested expert widths) with a dynamic Top-K router. Signature/version live in `config.py` (`Config.signature`, `Config.version`).

## Commands

Training (multi-GPU, configured for 2 devices in `train.sh`):
```
bash train.sh                              # CUDA_VISIBLE_DEVICES=0,1 python train_kl_k.py
```

Data prep (run once before training; produces TFDS shards under `artifacts/materials/data/`):
```
python prepare_train_data.py
```
`data01.py` is the older preprocessing variant; `prepare_train_data.py` is the current one (packed sequences).

Evaluation (BLEU + COMET on PhoMT test split):
```
python evaluate.py --checkpoint-step <step> --num-beams 1 --batch-size 64
python evaluate.py --skip-comet --limit 200          # quick sanity check
```

Parameter counting across `(elastic_mlp_dim, top_k)` combinations:
```
python count_params.py
```

There is no test suite, linter, or build step — this is a research training repo run directly with `python`.

## Architecture

### Two model variants (keep them in sync conceptually)
- `moe_model.py` + `moe_inference.py`: fixed Top-K MoE baseline.
- `moe_model_dynamic_k.py` + `moe_inference_dynamic_k.py`: **production** variant. Adds dynamic Top-K (sampled per step from `config.elastic_top_ks` / `elastic_top_k_probs`) on top of the Matryoshka elastic MLP.
- `train_kl_k.py` imports the `_dynamic_k` variants. The non-dynamic files are kept for ablations / reference — when editing model internals, update both unless intentionally diverging.

### Model layout (`moe_model_dynamic_k.py`)
- `MultiHeadAttention`: RoPE (`rotate_half` + precomputed sin/cos), QK-RMSNorm, no bias.
- `Expert`: gated MLP (SwiGLU-style) with `elastic_mlp_dims` — at each step a single `current_mlp_dim` is selected and only that prefix slice of weights is used (Matryoshka).
- `TaskConditionedSharedExpert`: dense shared expert conditioned on translation direction (`<translate-en-vi>` vs `<translate-vi-en>`, derived from the leading direction token id `vi_en_token_id`).
- `MoELayer`: routes tokens to `top_k` of `num_experts` experts; supports `current_top_k=0` (skip routed experts, shared expert only). Returns `(output, aux_load_balance_loss)`.
- `EncoderBlock` / `DecoderBlock` → `Encoder` / `Decoder` → `MoETranslationModel` (shared embedding for input/output, tied lm_head).
- `MoEModelConfig` (`flax.struct.dataclass` in `config.py`) is the static config passed through `nnx.jit` — changing its fields invalidates the jit cache.

### Training loop (`train_kl_k.py`)
- JAX `Mesh` over `('data',)` for data parallel; params replicated, batch sharded.
- Gradient accumulation via `optax.MultiSteps` (`config.grad_accum_steps`); effective batch = `batch_size * grad_accum_steps`.
- Loss = label-smoothed cross-entropy + MoE load-balance aux loss + **R-Drop KL** between two elastic MLP widths in the same step (the "KL_k" in the filename). `config.stop_gradient_kl` controls whether the larger-width branch detaches gradients. Encoder MSE term aligns encoder representations across widths.
- Per-step the trainer samples `current_mlp_dim` from `elastic_mlp_dims/probs` and `current_top_k` from `elastic_top_ks/probs` (with warmup/transition controlled by `k_warmup_steps`, `k_transition_steps`).
- LR: warmup → cosine decay (`warmup_steps`, `decay_steps`, `decay_end_value`).
- Checkpointing via Orbax to `config.checkpoint_path`; TensorBoard writes to `config.tensorboard_log_path` with separate writers for `train_avg`, `train_en2vi`, `train_vi2en`, `eval`, `test`. Previews and eval/test BLEU run every `*_interval` steps.

### Inference (`moe_inference_dynamic_k.py`)
- `generate_fast_greedy_jitted` and `generate_fast_beam_jitted` are `nnx.jit`'d with `current_mlp_dim` and `current_top_k` as **static** args — calling with a new combination triggers a recompile.
- KV cache is preallocated to `max_len` per call.
- Translation direction is inferred from the first token id of `source_ids` (matched against `cfg.vi_en_token_id`).

### Config (`config.py`)
- Single source of truth. `Config` is the runtime/training config (paths, schedule, intervals); `MoEModelConfig` is the static model config baked into jit. Most paths under `artifacts/...` are computed in `__post_init__` from `signature`, `version`, and input/output lengths.
- Tokenizer is loaded from `tokenizer_path_padded` (vocab padded to a multiple matching the device count — currently hardcoded to `_padded_8`; mismatch with `num_accelerator` is a known foot-gun).
- `new_special_tokens` includes the two direction markers `<translate-en-vi>` / `<translate-vi-en>` — these must be the first token of every input, both at training and inference time.

## Conventions / Gotchas
- `sys.path.extend(['/mnt/data/edw_2'])` appears in several entry points — there is an external dependency at that path on the training machine.
- `tf.config.set_visible_devices([], 'GPU')` is set everywhere so TF (used only for TFDS + TensorBoard) doesn't grab GPU memory away from JAX.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` is set high (0.985 train / 0.90 eval). Lower it locally if you OOM at startup.
- When changing `elastic_mlp_dims`, `num_experts`, or `top_k` candidates, both the probability lists in `Config` and the jit-static args in the inference functions need to be kept consistent.
