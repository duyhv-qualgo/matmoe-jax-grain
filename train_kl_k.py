import os
import sys
import time
import dataclasses
import threading
from typing import Any
import sacrebleu

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
sys.path.extend(['/mnt/data/edw_2'])

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import nnx
import flax.serialization
import optax
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import orbax.checkpoint as ocp
import msgpack
import grain
from datasets import load_from_disk

from config import config, MoEModelConfig
from moe_model_dynamic_k import MoETranslationModel
from moe_inference_dynamic_k import MoEGenerator, generate_fast_greedy_jitted, generate_fast_beam_jitted

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')


def main():
    print(f"\n{'-' * 60}\n 🚀 --- INITIALIZING SYSTEM & HARDWARE --- \n{'-' * 60}")
    print(f"[\033[92mInit\033[0m] Detected JAX Devices: \033[96m{jax.devices()}\033[0m")

    # Grad accum steps fallback to 4 if not specified in config
    grad_accum_steps = getattr(config, 'grad_accum_steps', 4)
    print(f"[\033[92mInit\033[0m] Gradient Accumulation Steps: \033[96m{grad_accum_steps}\033[0m")
    print(f"[\033[92mInit\033[0m] Effective Batch Size: \033[96m{config.batch_size * grad_accum_steps}\033[0m")

    # OVERLAY PLOT FIX: 3 distinct writers mapping to the same metric tag
    train_writer = tf.summary.create_file_writer(str(config.tensorboard_log_path / 'train_avg'))
    train_en2vi_writer = tf.summary.create_file_writer(str(config.tensorboard_log_path / 'train_en2vi'))
    train_vi2en_writer = tf.summary.create_file_writer(str(config.tensorboard_log_path / 'train_vi2en'))

    eval_writer = tf.summary.create_file_writer(str(config.tensorboard_log_path / 'eval'))
    test_writer = tf.summary.create_file_writer(str(config.tensorboard_log_path / 'test'))

    print(f"[\033[92mInit\033[0m] Loading Tokenizer from \033[96m{config.tokenizer_path_padded.name}\033[0m...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path_padded)
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vi_en_token_id = tokenizer.convert_tokens_to_ids("<translate-vi-en>")

    model_config = MoEModelConfig(
        vocab_size=vocab_size, pad_token_id=pad_token_id, vi_en_token_id=vi_en_token_id,
        d_model=config.d_model, num_heads=config.num_heads, mlp_dim=config.d_ff,
        num_layers=config.num_layers, num_experts=config.num_experts, top_k=config.top_k,
        semantic_dim=config.semantic_dim, dropout_rate=config.dropout_rate,
        max_seq_len=config.train_max_length_input, dtype=jnp.bfloat16
    )

    print(f"\n{'-' * 60}\n 💾 --- DATA PIPELINE --- \n{'-' * 60}")
    print(f"[\033[94mData\033[0m] Loading HF Arrow splits from base: \033[93m{config.dataset_path}\033[0m")

    hf_train = load_from_disk(config.dataset_path.as_posix() + "_train").with_format("numpy")
    train_ds = (
        grain.MapDataset.source(hf_train)
            .seed(config.seed)
            .shuffle()
            .repeat()
            .batch(config.batch_size * grad_accum_steps, drop_remainder=True)
            .to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=8))
            .mp_prefetch(grain.MultiprocessingOptions(num_workers=2))
    )

    def make_eval_ds(base_path: str, batch_size: int):
        hf = load_from_disk(base_path).with_format("numpy")
        return (grain.MapDataset.source(hf)
                  .batch(batch_size, drop_remainder=False)
                  .to_iter_dataset(grain.ReadOptions(num_threads=4, prefetch_buffer_size=4)))

    eval_ds_en_vi = make_eval_ds(config.dataset_path.as_posix() + "_validation_en_vi", config.eval_batch_size)
    eval_ds_vi_en = make_eval_ds(config.dataset_path.as_posix() + "_validation_vi_en", config.eval_batch_size)
    test_ds_en_vi = make_eval_ds(config.dataset_path.as_posix() + "_test_en_vi", config.test_batch_size)
    test_ds_vi_en = make_eval_ds(config.dataset_path.as_posix() + "_test_vi_en", config.test_batch_size)

    ds_iter = iter(train_ds)

    def to_macro(arr):
        # [B*G, S] -> [G, B, S]
        return arr.reshape(grad_accum_steps, config.batch_size, -1)

    steps_per_epoch = max(1, config.total_examples // (config.batch_size * grad_accum_steps))

    rngs = nnx.Rngs(config.seed)
    model = MoETranslationModel(model_config, rngs=rngs)
    generator = MoEGenerator(model, tokenizer, max_input_len=config.train_max_length_input)

    _, params, _ = nnx.split(model, nnx.Param, ...)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    print(f"\n{'-' * 60}\n 🧠 --- MODEL ARCHITECTURE --- \n{'-' * 60}")
    print(f" >>> TOTAL MODEL PARAMETERS: \033[1m\033[92m{total_params:,}\033[0m <<<")

    dtype_stats = {}
    flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
    printed_router, printed_expert, printed_embed, printed_tcse, printed_norm = False, False, False, False, False

    print(f"\n >>> MODEL DTYPE VERIFICATION <<<")
    for path, val in flat_params:
        path_str = jax.tree_util.keystr(path)
        dt = str(val.dtype)
        dtype_stats[dt] = dtype_stats.get(dt, 0) + val.size

        if "router" in path_str and "kernel" in path_str and not printed_router:
            color = "\033[93m" if "float32" in dt else "\033[91m"
            print(f"  🎯 Semantic Router -> {color}{dt}\033[0m")
            printed_router = True
        elif "shared_expert" in path_str and "w1" in path_str and not printed_expert:
            color = "\033[92m" if "bfloat16" in dt else "\033[91m"
            print(f"  🎯 Expert Weight   -> {color}{dt}\033[0m")
            printed_expert = True
        elif "embedding" in path_str and "embedding" in path_str and not printed_embed:
            color = "\033[92m" if "bfloat16" in dt else "\033[91m"
            print(f"  🎯 Vocab Embedding -> {color}{dt}\033[0m")
            printed_embed = True
        elif "shared_expert" in path_str and "gamma" in path_str and not printed_tcse:
            color = "\033[92m" if "bfloat16" in dt else "\033[91m"
            print(f"  🎯 TCSE Gamma Bias -> {color}{dt}\033[0m")
            printed_tcse = True
        elif "norm" in path_str and "scale" in path_str and not printed_norm:
            color = "\033[93m" if "float32" in dt else "\033[91m"
            print(f"  🎯 RMS Norms       -> {color}{dt}\033[0m")
            printed_norm = True

    print(f"\n 📊 \033[1mParameter Distribution by DType:\033[0m")
    for dt, count in dtype_stats.items():
        print(f"    - {dt:<10}: {count:>12,} params ({(count / total_params) * 100:.1f}%)")

    device_mesh = Mesh(np.array(jax.devices()), axis_names=('batch',))
    
    train_dp_sharding = NamedSharding(device_mesh, PartitionSpec(None, 'batch'))
    eval_dp_sharding = NamedSharding(device_mesh, PartitionSpec('batch', ))
    replicated_sharding = NamedSharding(device_mesh, PartitionSpec())

    print(f"\n📈 [\033[96mLR Schedule\033[0m] Initializing \033[93m{config.lr_schedule_type}\033[0m decay plan...")
    print(f"📊 [\033[96mTraining Info\033[0m] 1 Epoch = \033[93m{steps_per_epoch:,}\033[0m Macro-steps (Updates)")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.decay_init_value,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        end_value=config.decay_end_value
    )

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=0.01))
    
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    checkpointer = ocp.StandardCheckpointer()
    options = ocp.CheckpointManagerOptions(create=True, enable_async_checkpointing=True,
                                           max_to_keep=config.max_checkpoints_to_keep)
    ckpt_manager = ocp.CheckpointManager(str(config.checkpoint_path.resolve()), checkpointer, options)

    global_step = ckpt_manager.latest_step() or 0
    if global_step > 0:
        print(f"\n💾 [\033[93mCheckpoint\033[0m] Restoring from Update Step \033[92m{global_step}\033[0m...")

        data_ckpt_path = config.checkpoint_path / f"data_iter_{global_step}.msgpack"
        if data_ckpt_path.exists():
            with open(data_ckpt_path, "rb") as f:
                ds_iter.set_state(msgpack.unpackb(f.read(), raw=False))
            ds_iter.start_prefetch()
            print(f"💾 [\033[93mData\033[0m] Resumed iterator from {data_ckpt_path.name}")
        else:
            print(f"⚠️  No data_iter_{global_step}.msgpack — iterator restarts fresh")

        _, abstract_model_state = nnx.split(model)
        _, abstract_opt_state = nnx.split(optimizer)
        restored = ckpt_manager.restore(global_step, args=ocp.args.StandardRestore(
            item={'model': abstract_model_state, 'opt': abstract_opt_state}))
        nnx.update(model, restored['model'])
        nnx.update(optimizer, restored['opt'])

    print("⚙️  [\033[94mSharding\033[0m] Replicating weights across all devices...")
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)
    model_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, replicated_sharding), model_state)
    opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, replicated_sharding), opt_state)
    nnx.update(model, model_state)
    nnx.update(optimizer, opt_state)

    num_gpus = len(jax.devices())
    gpu_list_md = "\n".join([f"- **GPU {i}**: `{d.device_kind}`" for i, d in enumerate(jax.devices())])

    with train_writer.as_default():
        md_table = "| Key | Value |\n|---|---|\n"
        for field in dataclasses.fields(config):
            if field.name == 'preview_texts': continue
            val = getattr(config, field.name)
            val_str = ", ".join([str(v) for v in val]) if isinstance(val, list) else str(val)
            val_str = val_str.replace('<', '&lt;').replace('>', '&gt;')
            md_table += f"| **{field.name}** | {val_str} |\n"

        metadata_md = (
            f"### Experiment: {config.signature} (v{config.version})\n\n"
            f"**Dataset:** {config.dataset_name}\n\n"
            f"### Model Architecture\n\n"
            f"- **Total Parameters:** `{total_params:,}`\n"
            f"- **Upgrades:** ZERO-MALLOC Label Smoothing, R-Drop, Background Async Eval, Single Cosine LR, IN-JIT Grad Accum ({grad_accum_steps} steps)\n\n"
            f"### Hardware Environment\n\n"
            f"- **Total Accelerators:** {num_gpus}\n"
            f"{gpu_list_md}\n\n"
            f"### Configuration\n\n{md_table}"
        )
        tf.summary.text("System/Experiment_Metadata", metadata_md, step=global_step)

    def memory_efficient_cross_entropy(logits, labels, alpha=0.1):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        mean_log_probs = jnp.mean(log_probs, axis=-1)
        return -((1.0 - alpha) * target_log_probs + alpha * mean_log_probs)

    @nnx.jit(static_argnums=(5, 6))  
    # @nnx.jit()
    def macro_train_step(model, optimizer, macro_source_ids, macro_src_mask, macro_target_ids, dim1, dim2, k1, k2):
        graphdef, state = nnx.split(model)

        def loss_fn(m_model, c_source_ids, c_src_mask, c_target_ids):
            batch_sz = c_target_ids.shape[0]
            start_tokens = jnp.full((batch_sz, 1), model_config.pad_token_id, dtype=jnp.int32)
            decoder_input = jnp.concatenate([start_tokens, c_target_ids[:, :-1]], axis=1)
            labels = c_target_ids

            pad_mask = (decoder_input != model_config.pad_token_id).astype(jnp.int32).at[:, 0].set(1)
            causal_mask = jnp.tril(jnp.ones((decoder_input.shape[1], decoder_input.shape[1]), dtype=jnp.int32))
            tgt_mask = pad_mask[:, None, :] * causal_mask[None, :, :]

            logits1, aux_loss1, router_z_loss1 = m_model(c_source_ids, decoder_input, c_src_mask, tgt_mask,
                                                       current_mlp_dim=dim1, current_top_k=k1, deterministic=False)

            logits2, aux_loss2, router_z_loss2 = m_model(c_source_ids, decoder_input, c_src_mask, tgt_mask,
                                                       current_mlp_dim=dim2, current_top_k=k2, deterministic=False)

            loss_mask = (labels != model_config.pad_token_id).astype(jnp.float32)

            raw_loss1 = memory_efficient_cross_entropy(logits1, labels, alpha=0.1)
            raw_loss2 = memory_efficient_cross_entropy(logits2, labels, alpha=0.1)

            is_vi_en = (c_source_ids[:, 0] == vi_en_token_id).astype(jnp.float32)
            is_en_vi = 1.0 - is_vi_en
            mask_vi_en = loss_mask * is_vi_en[:, None]
            mask_en_vi = loss_mask * is_en_vi[:, None]

            def get_balanced_loss(raw_loss):
                loss_ve = jnp.sum(raw_loss * mask_vi_en) / jnp.maximum(jnp.sum(mask_vi_en), 1e-9)
                loss_ev = jnp.sum(raw_loss * mask_en_vi) / jnp.maximum(jnp.sum(mask_en_vi), 1e-9)
                has_ve = (jnp.sum(mask_vi_en) > 0.0).astype(jnp.float32)
                has_ev = (jnp.sum(mask_en_vi) > 0.0).astype(jnp.float32)
                total_active_directions = jnp.maximum(has_ve + has_ev, 1.0)
                return (loss_ve * has_ve + loss_ev * has_ev) / total_active_directions, loss_ve, loss_ev

            balanced_main1, loss_ve1, loss_ev1 = get_balanced_loss(raw_loss1)
            balanced_main2, loss_ve2, loss_ev2 = get_balanced_loss(raw_loss2)

            main_loss = (balanced_main1 + balanced_main2) / 2.0

            log_p1 = jax.nn.log_softmax(logits1, axis=-1)
            log_p2 = jax.nn.log_softmax(logits2, axis=-1)
            p1 = jnp.exp(log_p1)
            p2 = jnp.exp(log_p2)
            stop_gradient_kl = getattr(config, 'stop_gradient_kl', True)
            if stop_gradient_kl:
                sg = jax.lax.stop_gradient
                # Stop gradient on the 'target' distribution for each directional KL
                kl_1_to_2 = jnp.sum(sg(p1) * (sg(log_p1) - log_p2), axis=-1)
                kl_2_to_1 = jnp.sum(sg(p2) * (sg(log_p2) - log_p1), axis=-1)
                kl_div = 0.5 * (kl_1_to_2 + kl_2_to_1)
            else:
                kl_div = 0.5 * (jnp.sum(p1 * (log_p1 - log_p2), axis=-1) + jnp.sum(p2 * (log_p2 - log_p1), axis=-1))
            
            active_tokens = jnp.maximum(jnp.sum(loss_mask), 1e-9)
            kl_loss = jnp.sum(kl_div * loss_mask) / active_tokens

            aux_loss = (aux_loss1 + aux_loss2) / 2.0
            router_z_loss = (router_z_loss1 + router_z_loss2) / 2.0

            total_loss = main_loss + (0.01 * aux_loss) + (0.0001 * router_z_loss) + (1.0 * kl_loss)
            
            scaled_loss = total_loss / grad_accum_steps

            metrics_array = jnp.stack([
                main_loss.astype(jnp.float32), ((loss_ev1 + loss_ev2) / 2.0).astype(jnp.float32),
                ((loss_ve1 + loss_ve2) / 2.0).astype(jnp.float32), aux_loss.astype(jnp.float32),
                router_z_loss.astype(jnp.float32), kl_loss.astype(jnp.float32)
            ])
            return scaled_loss, (total_loss, metrics_array)

        def scan_fn(carry, inputs):
            accum_grads, accum_loss, accum_metrics, current_state = carry
            micro_s_ids, micro_s_mask, micro_t_ids = inputs
            
            temp_model = nnx.merge(graphdef, current_state)
            
            (scaled_loss, (total_loss, metrics_array)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(temp_model, micro_s_ids, micro_s_mask, micro_t_ids)
            
            _, new_state = nnx.split(temp_model)
            
            new_accum_grads = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
            new_accum_loss = accum_loss + total_loss / grad_accum_steps
            new_accum_metrics = accum_metrics + metrics_array / grad_accum_steps
            
            return (new_accum_grads, new_accum_loss, new_accum_metrics, new_state), None

        _, params, _ = nnx.split(model, nnx.Param, ...)
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        zero_loss = jnp.array(0.0, dtype=jnp.float32)
        zero_metrics = jnp.zeros((6,), dtype=jnp.float32)

        init_carry = (zero_grads, zero_loss, zero_metrics, state)

        final_carry, _ = jax.lax.scan(
            scan_fn,
            init_carry,
            (macro_source_ids, macro_src_mask, macro_target_ids)
        )

        final_grads, final_loss, final_metrics, final_state = final_carry

        nnx.update(model, final_state)
        optimizer.update(final_grads)
        
        return final_loss, final_metrics

    all_dims = config.elastic_mlp_dims
    dim_weights = config.elastic_mlp_probs
    
    all_top_ks = getattr(config, 'elastic_top_ks', [0, 1, 2])
    top_k_weights = getattr(config, 'elastic_top_k_probs', [1/3, 1/3, 1/3])
    
    print(f"\n  Matryoshka dims: {dict(zip(all_dims, dim_weights))}")
    print(f"  Dynamic top-k: {dict(zip(all_top_ks, top_k_weights))}")
    print(f"  Using single JIT function with dynamic dims to avoid recompilation")

    print(f"\n🔥 [\033[93mWarmup\033[0m] Pre-compiling JIT kernels for dynamic training...")
    
    warmup_batch = next(ds_iter)
    warmup_en = jax.device_put(to_macro(warmup_batch['input_ids']), train_dp_sharding)
    warmup_mask = jax.device_put(to_macro(warmup_batch['attention_mask']), train_dp_sharding)
    warmup_vi = jax.device_put(to_macro(warmup_batch['labels']), train_dp_sharding)
    
    # warmup_combinations = [(1024, 1024)]
    warmup_combinations = []
    for dim1 in all_dims:
        for dim2 in all_dims:
            warmup_combinations.append((dim1, dim2))
    
    total_combinations = len(warmup_combinations)
    print(f"   Reduced compile combinations to: {total_combinations}")
    warmup_start = time.time()
    
    # We pass dummy JAX tensors for Top-K since it is now entirely dynamic!
    dummy_k1 = jnp.array(all_top_ks[-1], dtype=jnp.int32)
    dummy_k2 = jnp.array(all_top_ks[-1], dtype=jnp.int32)
    
    with tqdm(total=total_combinations, desc="JIT Warmup", unit="combo", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as warmup_pbar:
        for dim1, dim2 in warmup_combinations:
            combo_start = time.time()
            
            loss_out, metrics_out = macro_train_step(model, optimizer, warmup_en, warmup_mask, warmup_vi, 
                                                     dim1, dim2, dummy_k1, dummy_k2)
            
            jax.block_until_ready(loss_out)
            jax.block_until_ready(metrics_out)
            
            combo_time = time.time() - combo_start
            warmup_pbar.set_postfix(dim1=dim1, dim2=dim2, time=f"{combo_time:.1f}s")
            warmup_pbar.update(1)
    
    warmup_elapsed = time.time() - warmup_start
    print(f"✅ [\033[92mWarmup Complete\033[0m] Compiled {total_combinations} combinations in {warmup_elapsed:.1f}s")

    def eval_step_with_dim(model, source_ids, src_mask, target_ids, dim, top_k=None):
        batch_sz = target_ids.shape[0]
        start_tokens = jnp.full((batch_sz, 1), model_config.pad_token_id, dtype=jnp.int32)
        decoder_input = jnp.concatenate([start_tokens, target_ids[:, :-1]], axis=1)

        pad_mask = (decoder_input != model_config.pad_token_id).astype(jnp.int32).at[:, 0].set(1)
        causal_mask = jnp.tril(jnp.ones((decoder_input.shape[1], decoder_input.shape[1]), dtype=jnp.int32))
        tgt_mask = pad_mask[:, None, :] * causal_mask[None, :, :]

        k_value = top_k if top_k is not None else config.top_k
        logits, _, _ = model(source_ids, decoder_input, src_mask, tgt_mask, current_mlp_dim=dim, 
                           current_top_k=k_value, deterministic=True)

        raw_loss = memory_efficient_cross_entropy(logits.astype(jnp.float32), target_ids, alpha=0.1)
        loss_mask = (target_ids != model_config.pad_token_id).astype(jnp.float32)

        return jnp.sum(raw_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1e-9)

    print(f"\n{'-' * 60}\n 🔥 --- STARTING MATRYOSHKA MOE TRAINING --- 🔥 \n{'-' * 60}")
    last_time = time.time()
    preview_texts = config.preview_texts
    
    import random

    # ========================================================
    # 🚀 SMOOTH K-CURRICULUM CONFIGURATION
    # ========================================================
    K_WARMUP_STEPS = getattr(config, 'k_warmup_steps', 20000)
    K_TRANSITION_STEPS = getattr(config, 'k_transition_steps', 10000) # Linearly blend probabilities over 10k steps
    print(f"  [\033[93mCurriculum\033[0m] Strict Warmup (k={config.top_k}): \033[96m{K_WARMUP_STEPS:,}\033[0m steps.")
    print(f"  [\033[93mCurriculum\033[0m] Linear Blending Transition: \033[96m{K_TRANSITION_STEPS:,}\033[0m steps.")

    try:
     # === BENCHMARK ACCUMULATORS (per log_interval window) ===
     bench_t_data = 0.0
     bench_t_h2d = 0.0
     bench_t_step = 0.0
     dataloader_name = "grain"
     with tqdm(total=config.total_training_steps, initial=global_step, desc="MatMoE Training (Updates)", unit="upd") as pbar:
        while global_step < config.total_training_steps:

            _t0 = time.perf_counter()
            batch = next(ds_iter)
            _t1 = time.perf_counter()
            bench_t_data += (_t1 - _t0)

            sharded_en = jax.device_put(to_macro(batch['input_ids']), train_dp_sharding)
            sharded_mask = jax.device_put(to_macro(batch['attention_mask']), train_dp_sharding)
            sharded_vi = jax.device_put(to_macro(batch['labels']), train_dp_sharding)
            _t2 = time.perf_counter()
            bench_t_h2d += (_t2 - _t1)

            py_dim1, py_dim2 = random.sample(all_dims, k=2)
            
            # ========================================================
            # 🚀 DYNAMIC K LINEAR CURRICULUM EXECUTION
            # ========================================================
            if global_step < K_WARMUP_STEPS:
                # Phase 1: Force strictly max k (config.top_k) during warmup
                py_k1, py_k2 = config.top_k, config.top_k
            elif global_step < K_WARMUP_STEPS + K_TRANSITION_STEPS:
                # Phase 2: Smooth linear interpolation of probabilities
                progress = (global_step - K_WARMUP_STEPS) / K_TRANSITION_STEPS
                
                current_weights = []
                for i, k_val in enumerate[Any](all_top_ks):
                    target_w = top_k_weights[i]
                    start_w = 1.0 if k_val == config.top_k else 0.0
                    curr_w = start_w + progress * (target_w - start_w) # Linear blend
                    current_weights.append(curr_w)
                    
                py_k1, py_k2 = random.choices(all_top_ks, weights=current_weights, k=2)
            else:
                # Phase 3: Normal Dynamic K probabilities after transition
                py_k1, py_k2 = random.choices(all_top_ks, weights=top_k_weights, k=2)
            
            k1_tensor = jnp.array(py_k1, dtype=jnp.int32)
            k2_tensor = jnp.array(py_k2, dtype=jnp.int32)
            
            _t3 = time.perf_counter()
            t_loss, metrics_array = macro_train_step(model, optimizer, sharded_en, sharded_mask, sharded_vi,
                                                    py_dim1, py_dim2, k1_tensor, k2_tensor)
            jax.block_until_ready(t_loss)
            _t4 = time.perf_counter()
            bench_t_step += (_t4 - _t3)

            global_step += 1
            pbar.update(1)

            t_loss_float = float(jax.device_get(t_loss))
            metrics_cpu = jax.device_get(metrics_array)
            m_loss, loss_ev, loss_ve, a_loss, rz_loss, kl_loss = [float(x) for x in metrics_cpu]

            if global_step % config.log_interval == 0:
                current_time = time.time()

                samples_processed = config.batch_size * grad_accum_steps * config.log_interval
                speed_samples = samples_processed / (current_time - last_time)

                # === BENCHMARK REPORT (avg ms per macro-step over the window) ===
                _n = max(config.log_interval, 1)
                ms_data = bench_t_data / _n * 1000
                ms_h2d  = bench_t_h2d  / _n * 1000
                ms_step = bench_t_step / _n * 1000
                ms_total = ms_data + ms_h2d + ms_step
                data_pct = (bench_t_data / max(bench_t_data + bench_t_h2d + bench_t_step, 1e-9)) * 100
                print(f"\n[BENCH/{dataloader_name}] step={global_step} "
                      f"t_data={ms_data:.1f}ms  t_h2d={ms_h2d:.1f}ms  t_step={ms_step:.1f}ms  "
                      f"total={ms_total:.1f}ms  data_share={data_pct:.1f}%  sps={speed_samples:.0f}")
                bench_t_data = 0.0
                bench_t_h2d = 0.0
                bench_t_step = 0.0

                current_lr = float(lr_schedule(global_step))
                current_epoch_float = global_step / steps_per_epoch

                pbar.set_postfix(
                    Avg=f"{m_loss:.2f}", KL=f"{kl_loss:.3f}",
                    dims=f"{py_dim1}/{py_dim2}", ks=f"{py_k1}/{py_k2}",
                    lr=f"{current_lr:.2e}", sps=f"{speed_samples:.0f}"
                )

                with train_writer.as_default():
                    tf.summary.scalar('Loss/Translation_Overlay', m_loss, step=global_step)
                    tf.summary.scalar('Loss/Total', t_loss_float, step=global_step)
                    tf.summary.scalar('Loss/Routing_Penalty', a_loss, step=global_step)
                    tf.summary.scalar('Loss/Router_Z_Loss', rz_loss, step=global_step)
                    tf.summary.scalar('Loss/KL_Between_Dims', kl_loss, step=global_step)
                    tf.summary.scalar('System/Learning_Rate', current_lr, step=global_step)
                    tf.summary.scalar('System/Speed_SPS', speed_samples, step=global_step)
                    tf.summary.scalar('System/Epoch', current_epoch_float, step=global_step)
                    tf.summary.scalar('System/Dim1', float(py_dim1), step=global_step)
                    tf.summary.scalar('System/Dim2', float(py_dim2), step=global_step)
                    tf.summary.scalar('System/TopK1', float(py_k1), step=global_step)
                    tf.summary.scalar('System/TopK2', float(py_k2), step=global_step)
                train_writer.flush()

                with train_en2vi_writer.as_default():
                    tf.summary.scalar('Loss/Translation_Overlay', loss_ev, step=global_step)
                train_en2vi_writer.flush()

                with train_vi2en_writer.as_default():
                    tf.summary.scalar('Loss/Translation_Overlay', loss_ve, step=global_step)
                train_vi2en_writer.flush()

                last_time = current_time

            if global_step % config.eval_interval == 0:
                def run_eval_multi_dim(ds):
                    all_losses = []
                    for b in ds:
                        in_np, mask_np, labels_np = b['input_ids'], b['attention_mask'], b['labels']
                        actual_bsz = in_np.shape[0]
                        if actual_bsz < config.eval_batch_size:
                            pad_amt = config.eval_batch_size - actual_bsz
                            in_np = np.pad(in_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)
                            mask_np = np.pad(mask_np, ((0, pad_amt), (0, 0)), constant_values=0)
                            labels_np = np.pad(labels_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)
                        
                        sharded_in = jax.device_put(in_np, eval_dp_sharding)
                        sharded_mask = jax.device_put(mask_np, eval_dp_sharding)
                        sharded_labels = jax.device_put(labels_np, eval_dp_sharding)
                        
                        batch_losses = []
                        for dim in all_dims:
                            batch_losses.append(float(eval_step_with_dim(model, sharded_in, sharded_mask, sharded_labels, dim)))
                        all_losses.append(batch_losses)
                    return np.mean(all_losses, axis=0)
                
                def run_eval_dim_1024_with_topk(ds):
                    """Evaluate dim=1024 with all top-k values"""
                    all_losses = {k: [] for k in all_top_ks}
                    for b in ds:
                        in_np, mask_np, labels_np = b['input_ids'], b['attention_mask'], b['labels']
                        actual_bsz = in_np.shape[0]
                        if actual_bsz < config.eval_batch_size:
                            pad_amt = config.eval_batch_size - actual_bsz
                            in_np = np.pad(in_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)
                            mask_np = np.pad(mask_np, ((0, pad_amt), (0, 0)), constant_values=0)
                            labels_np = np.pad(labels_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)
                        
                        sharded_in = jax.device_put(in_np, eval_dp_sharding)
                        sharded_mask = jax.device_put(mask_np, eval_dp_sharding)
                        sharded_labels = jax.device_put(labels_np, eval_dp_sharding)
                        
                        for k in all_top_ks:
                            loss = float(eval_step_with_dim(model, sharded_in, sharded_mask, sharded_labels, 1024, top_k=k))
                            all_losses[k].append(loss)
                    
                    return {k: np.mean(losses) for k, losses in all_losses.items()}

                eval_ev_metrics = run_eval_multi_dim(eval_ds_en_vi)
                eval_ve_metrics = run_eval_multi_dim(eval_ds_vi_en)
                
                eval_1024_ev_topk = run_eval_dim_1024_with_topk(eval_ds_en_vi)
                eval_1024_ve_topk = run_eval_dim_1024_with_topk(eval_ds_vi_en)
                
                print(f"\n  [\033[94mEval\033[0m] Update Step {global_step}")
                
                eval_results = []
                for idx, dim in enumerate(all_dims):
                    ev_loss = float(eval_ev_metrics[idx])
                    ve_loss = float(eval_ve_metrics[idx])
                    avg_eval = (ev_loss + ve_loss) / 2.0
                    eval_results.append((dim, ev_loss, ve_loss, avg_eval))
                    print(f"    dim={dim:>5}: en->vi={ev_loss:.4f}  vi->en={ve_loss:.4f}  avg={avg_eval:.4f}")
                
                print(f"  [\033[96mDim 1024 with Top-K variants\033[0m]")
                for k in all_top_ks:
                    ev_loss_k = eval_1024_ev_topk[k]
                    ve_loss_k = eval_1024_ve_topk[k]
                    avg_k = (ev_loss_k + ve_loss_k) / 2.0
                    print(f"    dim=1024, k={k}: en->vi={ev_loss_k:.4f}  vi->en={ve_loss_k:.4f}  avg={avg_k:.4f}")
                    
                with eval_writer.as_default():
                    for dim, ev_loss, ve_loss, avg_eval in eval_results:
                        tf.summary.scalar(f'Loss/Dim_{dim}_EnVi', ev_loss, step=global_step)
                        tf.summary.scalar(f'Loss/Dim_{dim}_ViEn', ve_loss, step=global_step)
                        tf.summary.scalar(f'Loss/Dim_{dim}_Avg', avg_eval, step=global_step)
                    
                    for k in all_top_ks:
                        ev_loss_k = eval_1024_ev_topk[k]
                        ve_loss_k = eval_1024_ve_topk[k]
                        avg_k = (ev_loss_k + ve_loss_k) / 2.0
                        tf.summary.scalar(f'Loss/Dim_1024_TopK_{k}_EnVi', ev_loss_k, step=global_step)
                        tf.summary.scalar(f'Loss/Dim_1024_TopK_{k}_ViEn', ve_loss_k, step=global_step)
                        tf.summary.scalar(f'Loss/Dim_1024_TopK_{k}_Avg', avg_k, step=global_step)
                
                eval_writer.flush()

            if global_step % config.preview_interval == 0:
                greedy_texts, g_metrics = generator.generate(preview_texts, method="greedy",
                                                             max_len=config.max_length_inference, verbose=False)
                sampled_texts, s_metrics = generator.generate(preview_texts, method="sample",
                                                              max_len=config.max_length_inference, temperature=0.8,
                                                              top_p=0.9, top_k=40, seed=global_step, verbose=False)
                beam_texts, b_metrics = generator.generate(preview_texts, method="beam",
                                                           max_len=config.max_length_inference, verbose=False)

                markdown_str = f"**Hardware Speeds** | Greedy: {g_metrics['tps_effective']:.1f} TPS | Beam(4): {b_metrics['tps_effective']:.1f} TPS | Sampled: {s_metrics['tps_effective']:.1f} TPS\n\n---\n\n"

                for i, src in enumerate(preview_texts):
                    markdown_str += f"**📥 Input:** {src.replace('<', '&lt;').replace('>', '&gt;')}  \n**🎯 Greedy:** {greedy_texts[i].replace('<', '&lt;').replace('>', '&gt;')}  \n**🚀 Beam(4):** {beam_texts[i].replace('<', '&lt;').replace('>', '&gt;')}  \n**🎲 Sampled(0.8):** {sampled_texts[i].replace('<', '&lt;').replace('>', '&gt;')}  \n\n---\n\n"
                with eval_writer.as_default():
                    tf.summary.text("Evaluation/Predictions", markdown_str, step=global_step)

            if global_step % config.test_interval == 0:
                import re
                print(f"\n\n🔬 [\033[93mTest\033[0m] Generating test batches for Update step \033[92m{global_step}\033[0m...")

                all_dim_predictions = {dim: {
                    'greedy_ev': [], 'beam_ev': [], 
                    'greedy_ve': [], 'beam_ve': []
                } for dim in all_dims}
                
                dim_1024_topk_predictions = {k: {
                    'greedy_ev': [], 'beam_ev': [], 
                    'greedy_ve': [], 'beam_ve': []
                } for k in all_top_ks}
                
                raw_r_ev, raw_r_ve = [], []

                eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

                def generate_test_set_multi_dim(ds, refs, is_en_vi=True):
                    for b_idx, b in enumerate(ds):
                        in_np, mask_np, labels_np = b['input_ids'], b['attention_mask'], b['labels']
                        actual_bsz = in_np.shape[0]

                        if actual_bsz < config.test_batch_size:
                            pad_amt = config.test_batch_size - actual_bsz
                            in_np = np.pad(in_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)
                            mask_np = np.pad(mask_np, ((0, pad_amt), (0, 0)), constant_values=0)
                            labels_np = np.pad(labels_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)

                        sharded_in = jax.device_put(in_np, eval_dp_sharding)
                        sharded_mask = jax.device_put(mask_np, eval_dp_sharding)

                        for dim in all_dims:
                            out_ids_greedy, _ = generate_fast_greedy_jitted(model, sharded_in, sharded_mask,
                                                                            config.max_length_inference, pad_token_id,
                                                                            eos_id, current_mlp_dim=dim)
                            out_ids_beam, _ = generate_fast_beam_jitted(model, sharded_in, sharded_mask,
                                                                        config.max_length_inference, pad_token_id, 
                                                                        eos_id, current_mlp_dim=dim)
                            
                            key_g = 'greedy_ev' if is_en_vi else 'greedy_ve'
                            key_b = 'beam_ev' if is_en_vi else 'beam_ve'
                            all_dim_predictions[dim][key_g].append(np.array(out_ids_greedy)[:actual_bsz])
                            all_dim_predictions[dim][key_b].append(np.array(out_ids_beam)[:actual_bsz])
                        
                        refs.append(labels_np[:actual_bsz])
                
                def generate_test_set_dim_1024_topk(ds, is_en_vi=True):
                    """Generate predictions for dim=1024 with all top-k values"""
                    for b_idx, b in enumerate(ds):
                        in_np, mask_np, labels_np = b['input_ids'], b['attention_mask'], b['labels']
                        actual_bsz = in_np.shape[0]

                        if actual_bsz < config.test_batch_size:
                            pad_amt = config.test_batch_size - actual_bsz
                            in_np = np.pad(in_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)
                            mask_np = np.pad(mask_np, ((0, pad_amt), (0, 0)), constant_values=0)
                            labels_np = np.pad(labels_np, ((0, pad_amt), (0, 0)), constant_values=pad_token_id)

                        sharded_in = jax.device_put(in_np, eval_dp_sharding)
                        sharded_mask = jax.device_put(mask_np, eval_dp_sharding)

                        for k in all_top_ks:
                            out_ids_greedy, _ = generate_fast_greedy_jitted(model, sharded_in, sharded_mask,
                                                                            config.max_length_inference, pad_token_id,
                                                                            eos_id, current_mlp_dim=1024, current_top_k=k)
                            out_ids_beam, _ = generate_fast_beam_jitted(model, sharded_in, sharded_mask,
                                                                        config.max_length_inference, pad_token_id, 
                                                                        eos_id, current_mlp_dim=1024, current_top_k=k)
                            
                            key_g = 'greedy_ev' if is_en_vi else 'greedy_ve'
                            key_b = 'beam_ev' if is_en_vi else 'beam_ve'
                            dim_1024_topk_predictions[k][key_g].append(np.array(out_ids_greedy)[:actual_bsz])
                            dim_1024_topk_predictions[k][key_b].append(np.array(out_ids_beam)[:actual_bsz])

                generate_test_set_multi_dim(test_ds_en_vi, raw_r_ev, is_en_vi=True)
                generate_test_set_multi_dim(test_ds_vi_en, raw_r_ve, is_en_vi=False)
                
                generate_test_set_dim_1024_topk(test_ds_en_vi, is_en_vi=True)
                generate_test_set_dim_1024_topk(test_ds_vi_en, is_en_vi=False)
                print(f"✅ [\033[92mGPU Free\033[0m] Handing off decoding to Background Thread...")

                def compute_metrics_async_multi_dim(step, dim_preds, dim_1024_topk_preds, r_ref_ev, r_ref_ve):
                    def clean_detokenize(texts):
                        return [re.sub(r'([a-zA-Z])(\d)', r'\1 \2', re.sub(r'([a-z])([A-Z])', r'\1 \2',
                                                                           t.replace(" .", ".").replace(" ,",
                                                                                                        ",").replace(
                                                                               " ?", "?").replace(" !", "!").replace(
                                                                               " :", ":").replace(" ;", ";"))).strip()
                                for t in texts]

                    def decode_and_clean(arrays):
                        flat = np.concatenate(arrays, axis=0) if arrays else np.array([])
                        if len(flat) == 0: return []
                        decoded = tokenizer.batch_decode(flat, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
                        return clean_detokenize(decoded)

                    def decode_refs(arrays):
                        flat = np.concatenate(arrays, axis=0) if arrays else np.array([])
                        if len(flat) == 0: return []
                        cleaned = np.where(flat != pad_token_id, flat, tokenizer.pad_token_id)
                        decoded = tokenizer.batch_decode(cleaned, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
                        return clean_detokenize(decoded)

                    ref_ev = decode_refs(r_ref_ev)
                    ref_ve = decode_refs(r_ref_ve)

                    print(f"\n🧵 [\033[94mBackground Computing BLEU\033[0m] Step \033[96m{step}\033[0m")
                    
                    for dim in all_dims:
                        p_g_ev = decode_and_clean(dim_preds[dim]['greedy_ev'])
                        p_b_ev = decode_and_clean(dim_preds[dim]['beam_ev'])
                        p_g_ve = decode_and_clean(dim_preds[dim]['greedy_ve'])
                        p_b_ve = decode_and_clean(dim_preds[dim]['beam_ve'])

                        b_g_ev = sacrebleu.corpus_bleu(p_g_ev, [ref_ev]).score if p_g_ev else 0.0
                        b_g_ve = sacrebleu.corpus_bleu(p_g_ve, [ref_ve]).score if p_g_ve else 0.0
                        b_g_avg = (b_g_ev + b_g_ve) / 2.0

                        b_b_ev = sacrebleu.corpus_bleu(p_b_ev, [ref_ev]).score if p_b_ev else 0.0
                        b_b_ve = sacrebleu.corpus_bleu(p_b_ve, [ref_ve]).score if p_b_ve else 0.0
                        b_b_avg = (b_b_ev + b_b_ve) / 2.0

                        with test_writer.as_default():
                            tf.summary.scalar(f'BLEU_Greedy/Dim_{dim}_Avg', b_g_avg, step=step)
                            tf.summary.scalar(f'BLEU_Greedy/Dim_{dim}_En2Vi', b_g_ev, step=step)
                            tf.summary.scalar(f'BLEU_Greedy/Dim_{dim}_Vi2En', b_g_ve, step=step)

                            tf.summary.scalar(f'BLEU_Beam4/Dim_{dim}_Avg', b_b_avg, step=step)
                            tf.summary.scalar(f'BLEU_Beam4/Dim_{dim}_En2Vi', b_b_ev, step=step)
                            tf.summary.scalar(f'BLEU_Beam4/Dim_{dim}_Vi2En', b_b_ve, step=step)

                        print(f"  dim={dim:>5}: Greedy={b_g_avg:.2f} (En:{b_g_ev:.2f}, Vi:{b_g_ve:.2f}) | Beam={b_b_avg:.2f} (En:{b_b_ev:.2f}, Vi:{b_b_ve:.2f})")
                    
                    print(f"  [\033[96mDim 1024 with Top-K variants\033[0m]")
                    for k in all_top_ks:
                        p_g_ev_k = decode_and_clean(dim_1024_topk_preds[k]['greedy_ev'])
                        p_b_ev_k = decode_and_clean(dim_1024_topk_preds[k]['beam_ev'])
                        p_g_ve_k = decode_and_clean(dim_1024_topk_preds[k]['greedy_ve'])
                        p_b_ve_k = decode_and_clean(dim_1024_topk_preds[k]['beam_ve'])

                        b_g_ev_k = sacrebleu.corpus_bleu(p_g_ev_k, [ref_ev]).score if p_g_ev_k else 0.0
                        b_g_ve_k = sacrebleu.corpus_bleu(p_g_ve_k, [ref_ve]).score if p_g_ve_k else 0.0
                        b_g_avg_k = (b_g_ev_k + b_g_ve_k) / 2.0

                        b_b_ev_k = sacrebleu.corpus_bleu(p_b_ev_k, [ref_ev]).score if p_b_ev_k else 0.0
                        b_b_ve_k = sacrebleu.corpus_bleu(p_b_ve_k, [ref_ve]).score if p_b_ve_k else 0.0
                        b_b_avg_k = (b_b_ev_k + b_b_ve_k) / 2.0

                        with test_writer.as_default():
                            tf.summary.scalar(f'BLEU_Greedy/Dim_1024_TopK_{k}_Avg', b_g_avg_k, step=step)
                            tf.summary.scalar(f'BLEU_Greedy/Dim_1024_TopK_{k}_En2Vi', b_g_ev_k, step=step)
                            tf.summary.scalar(f'BLEU_Greedy/Dim_1024_TopK_{k}_Vi2En', b_g_ve_k, step=step)

                            tf.summary.scalar(f'BLEU_Beam4/Dim_1024_TopK_{k}_Avg', b_b_avg_k, step=step)
                            tf.summary.scalar(f'BLEU_Beam4/Dim_1024_TopK_{k}_En2Vi', b_b_ev_k, step=step)
                            tf.summary.scalar(f'BLEU_Beam4/Dim_1024_TopK_{k}_Vi2En', b_b_ve_k, step=step)

                        print(f"  dim=1024, k={k}: Greedy={b_g_avg_k:.2f} (En:{b_g_ev_k:.2f}, Vi:{b_g_ve_k:.2f}) | Beam={b_b_avg_k:.2f} (En:{b_b_ev_k:.2f}, Vi:{b_b_ve_k:.2f})")

                    print(f"✅ [\033[92mBackground Done\033[0m]\n")

                threading.Thread(target=compute_metrics_async_multi_dim,
                                 args=(global_step, all_dim_predictions, dim_1024_topk_predictions, raw_r_ev, raw_r_ve)).start()

            if global_step % config.ckpt_interval == 0:
                _, m_st = nnx.split(model)
                _, o_st = nnx.split(optimizer)
                ckpt_manager.save(global_step, args=ocp.args.StandardSave({'model': m_st, 'opt': o_st}))

                data_ckpt_path = config.checkpoint_path / f"data_iter_{global_step}.msgpack"
                with open(data_ckpt_path, "wb") as f:
                    f.write(msgpack.packb(ds_iter.get_state(), use_bin_type=True))

            if global_step % config.model_save_interval == 0:
                _, model_params, _ = nnx.split(model, nnx.Param, ...)

                def unwrap_state(x):
                    if isinstance(x, (nnx.Variable, nnx.VariableState)):
                        return x.value
                    if hasattr(x, "items"):
                        return {k: unwrap_state(v) for k, v in x.items()}
                    return x

                with open(config.latest_msg_path, "wb") as f: f.write(
                    flax.serialization.msgpack_serialize(unwrap_state(model_params)))
    finally:
        try:
            ds_iter.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()