#!/usr/bin/env python3
"""Show stored and active parameter counts for all (dim, k) combinations."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import AutoTokenizer

from config import config as moe_config, MoEModelConfig
from moe_model_dynamic_k import MoETranslationModel

# ---------------------------------------------------------------------------
# Build model (random weights — only shapes matter for counting)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(moe_config.tokenizer_path_padded)
vocab_size = len(tokenizer)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
vi_en_token_id = tokenizer.convert_tokens_to_ids("<translate-vi-en>")

model_config = MoEModelConfig(
    vocab_size=vocab_size, pad_token_id=pad_token_id, vi_en_token_id=vi_en_token_id,
    d_model=moe_config.d_model, num_heads=moe_config.num_heads,
    mlp_dim=moe_config.d_ff, num_layers=moe_config.num_layers,
    num_experts=moe_config.num_experts, top_k=moe_config.top_k,
    semantic_dim=moe_config.semantic_dim,
    dropout_rate=0.0, max_seq_len=moe_config.train_max_length_input,
    dtype=jnp.bfloat16
)

rngs = nnx.Rngs(moe_config.seed)
model = MoETranslationModel(model_config, rngs=rngs)

total_params = sum(x.size for x in jax.tree.leaves(nnx.state(model)))


# ---------------------------------------------------------------------------
# Counting helpers (same logic as evaluate_multidim_multik.py)
# ---------------------------------------------------------------------------
def count_effective_params(model, mlp_dim):
    """Parameters stored / used when running at a given mlp_dim (Matryoshka slice)."""
    total = 0
    max_mlp_dim = model.cfg.mlp_dim
    for path, leaf in jax.tree.leaves_with_path(nnx.state(model)):
        path_str = "/".join(str(k) for k in path)
        shape = leaf.shape if hasattr(leaf, "shape") else ()
        if not shape:
            total += 1
            continue
        param_count = int(np.prod(shape))
        is_moe = "experts" in path_str or "shared_expert" in path_str
        if not is_moe:
            total += param_count
            continue
        if "w1" in path_str and "kernel" in path_str and len(shape) == 2:
            param_count = shape[0] * mlp_dim
        elif "w2" in path_str and "kernel" in path_str and len(shape) == 2:
            param_count = shape[0] * mlp_dim
        elif "w3" in path_str and "kernel" in path_str and len(shape) == 2:
            param_count = mlp_dim * shape[1]
        elif ("gamma" in path_str or "beta" in path_str) and len(shape) == 2:
            param_count = shape[0] * mlp_dim
        total += param_count
    return total


def count_active_params(model, mlp_dim, route_k=None):
    """Parameters actively used per token at (mlp_dim, route_k)."""
    top_k = min(int(route_k), model.cfg.top_k) if route_k is not None else model.cfg.top_k
    num_experts = model.cfg.num_experts
    total = 0
    expert_params = 0
    for path, leaf in jax.tree.leaves_with_path(nnx.state(model)):
        path_str = "/".join(str(k) for k in path)
        shape = leaf.shape if hasattr(leaf, "shape") else ()
        if not shape:
            total += 1
            continue
        param_count = int(np.prod(shape))
        is_routed_expert = "experts" in path_str and "shared_expert" not in path_str
        if not is_routed_expert:
            if "shared_expert" in path_str:
                if "w1" in path_str and "kernel" in path_str and len(shape) == 2:
                    param_count = shape[0] * mlp_dim
                elif "w2" in path_str and "kernel" in path_str and len(shape) == 2:
                    param_count = shape[0] * mlp_dim
                elif "w3" in path_str and "kernel" in path_str and len(shape) == 2:
                    param_count = mlp_dim * shape[1]
                elif ("gamma" in path_str or "beta" in path_str) and len(shape) == 2:
                    param_count = shape[0] * mlp_dim
            total += param_count
            continue
        if "w1" in path_str and "kernel" in path_str and len(shape) == 2:
            param_count = shape[0] * mlp_dim
        elif "w2" in path_str and "kernel" in path_str and len(shape) == 2:
            param_count = shape[0] * mlp_dim
        elif "w3" in path_str and "kernel" in path_str and len(shape) == 2:
            param_count = mlp_dim * shape[1]
        elif ("gamma" in path_str or "beta" in path_str) and len(shape) == 2:
            param_count = shape[0] * mlp_dim
        expert_params += param_count
    total += (expert_params * top_k) // num_experts
    return total


# ---------------------------------------------------------------------------
# Enumerate all dims and ks
# ---------------------------------------------------------------------------
all_dims = list(moe_config.elastic_mlp_dims)
all_ks   = list(range(int(model.cfg.top_k) + 1))  # 0 .. top_k inclusive

dim_stored  = {dim: count_effective_params(model, dim) for dim in all_dims}
dim_k_active = {dim: {k: count_active_params(model, dim, route_k=k) for k in all_ks}
                for dim in all_dims}

# ---------------------------------------------------------------------------
# Print architecture summary
# ---------------------------------------------------------------------------
print("=" * 72)
print("  MODEL ARCHITECTURE")
print("=" * 72)
print(f"  d_model:      {model.cfg.d_model}")
print(f"  mlp_dim (max):{model.cfg.mlp_dim}")
print(f"  num_layers:   {model.cfg.num_layers}")
print(f"  num_experts:  {model.cfg.num_experts}  (routed, top_k={model.cfg.top_k})")
print(f"  vocab_size:   {model.cfg.vocab_size}")
print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  Elastic dims: {all_dims}")
print(f"  Elastic top-ks: {all_ks}")
print("=" * 72)

# ---------------------------------------------------------------------------
# Param grid table: Stored + Active@k for every (dim, k)
# ---------------------------------------------------------------------------
col_w = 16
k_header = "".join(f"  {'Active@k='+str(k):>{col_w}}" for k in all_ks)
print(f"\n  {'Dim':>6}  {'Stored':>{col_w}}" + k_header)
print("  " + "-" * (6 + 2 + col_w + len(all_ks) * (col_w + 2) + 4))
for dim in all_dims:
    s = dim_stored[dim]
    k_cols = "".join(f"  {dim_k_active[dim][k]/1e6:>{col_w-1}.2f}M" for k in all_ks)
    print(f"  {dim:>6}  {s/1e6:>{col_w-1}.2f}M" + k_cols)
print()

# ---------------------------------------------------------------------------
# Active / Stored ratio table
# ---------------------------------------------------------------------------
print(f"  {'Dim':>6}  {'Stored':>{col_w}}" + "".join(f"  {'Ratio@k='+str(k):>{col_w}}" for k in all_ks))
print("  " + "-" * (6 + 2 + col_w + len(all_ks) * (col_w + 2) + 4))
for dim in all_dims:
    s = dim_stored[dim]
    k_cols = "".join(f"  {dim_k_active[dim][k]/s:>{col_w-1}.1%}" for k in all_ks)
    print(f"  {dim:>6}  {s/1e6:>{col_w-1}.2f}M" + k_cols)
print()

print(f"Note: top_k=0 means only the shared expert is active (no routed experts).")
print(f"      top_k={model.cfg.top_k} is the maximum (full model).")
