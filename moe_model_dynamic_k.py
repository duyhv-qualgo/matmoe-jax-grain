import jax
import jax.numpy as jnp
import jax.scipy.special
from flax import nnx
from config import MoEModelConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float, dtype: any, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)
        self.key = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)
        self.value = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)
        self.out = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)

        self.q_norm = nnx.RMSNorm(self.head_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, mask=None, context=None, sin=None, cos=None, deterministic: bool = False):
        batch_size, seq_len, _ = x.shape
        context = x if context is None else context
        ctx_len = context.shape[1]

        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(context).reshape(batch_size, ctx_len, self.num_heads, self.head_dim)
        v = self.value(context).reshape(batch_size, ctx_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if sin is not None and cos is not None:
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        attn_scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            attn_scores = jnp.where(mask == 1, attn_scores, jnp.array(-jnp.inf, dtype=attn_scores.dtype))

        attn_probs = jax.nn.softmax(attn_scores.astype(jnp.float32), axis=-1).astype(attn_scores.dtype)
        attn_probs = self.dropout(attn_probs, deterministic=deterministic)

        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_probs, v)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        return self.out(attn_output)


class Expert(nnx.Module):
    def __init__(self, d_model: int, max_mlp_dim: int, dropout_rate: float, dtype: any, rngs: nnx.Rngs):
        self.max_mlp_dim = max_mlp_dim
        self.w1 = nnx.Linear(d_model, max_mlp_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.w2 = nnx.Linear(d_model, max_mlp_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.w3 = nnx.Linear(max_mlp_dim, d_model, use_bias=False, dtype=dtype, rngs=rngs)
        
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, current_mlp_dim=None, deterministic: bool = False):
        dim = current_mlp_dim if current_mlp_dim is not None else self.max_mlp_dim
        w1_k = self.w1.kernel[:, :dim]
        w2_k = self.w2.kernel[:, :dim]
        w3_k = self.w3.kernel[:dim, :]

        scale_factor = jnp.sqrt(self.max_mlp_dim / dim)

        h = jax.nn.silu(jnp.dot(x, w1_k)) * jnp.dot(x, w2_k)
        h = self.dropout(h, deterministic=deterministic)
        return jnp.dot(h, w3_k) * scale_factor


class TaskConditionedSharedExpert(nnx.Module):
    def __init__(self, d_model: int, max_mlp_dim: int, num_tasks: int, dropout_rate: float, dtype: any, rngs: nnx.Rngs):
        self.max_mlp_dim = max_mlp_dim
        self.w1 = nnx.Linear(d_model, max_mlp_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.w2 = nnx.Linear(d_model, max_mlp_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.w3 = nnx.Linear(max_mlp_dim, d_model, use_bias=False, dtype=dtype, rngs=rngs)

        self.gamma = nnx.Param(jnp.zeros((num_tasks, max_mlp_dim), dtype=jnp.float32))
        self.beta = nnx.Param(jnp.zeros((num_tasks, max_mlp_dim), dtype=jnp.float32))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, direction_ids, current_mlp_dim=None, deterministic: bool = False):
        dim = current_mlp_dim if current_mlp_dim is not None else self.max_mlp_dim
        w1_k = self.w1.kernel[:, :dim]
        w2_k = self.w2.kernel[:, :dim]
        w3_k = self.w3.kernel[:dim, :]

        scale_factor = jnp.sqrt(self.max_mlp_dim / dim)

        g = self.gamma[...][direction_ids][:, None, :dim] if x.ndim == 3 else self.gamma[...][direction_ids][:, :dim]
        b = self.beta[...][direction_ids][:, None, :dim] if x.ndim == 3 else self.beta[...][direction_ids][:, :dim]

        h = jax.nn.silu(jnp.dot(x, w1_k)) * jnp.dot(x, w2_k)
        h = h * (1.0 + g) + b
        h = self.dropout(h, deterministic=deterministic)
        return jnp.dot(h, w3_k) * scale_factor


class MoELayer(nnx.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, mlp_dim: int, dropout_rate: float, dtype: any,
                 semantic_dim: int, rngs: nnx.Rngs):
        self.num_experts = num_experts
        self.top_k = top_k  # This acts as max_k

        self.task_embedding = nnx.Embed(2, semantic_dim, dtype=dtype, rngs=rngs)

        self.router = nnx.Linear(d_model + semantic_dim, num_experts, use_bias=False, dtype=jnp.float32,
                                 param_dtype=jnp.float32, rngs=rngs)

        self.experts = [Expert(d_model, mlp_dim, dropout_rate, dtype, rngs) for _ in range(num_experts)]
        self.shared_expert = TaskConditionedSharedExpert(d_model, mlp_dim, 2, dropout_rate, dtype, rngs)

    def __call__(self, x, direction_ids, current_mlp_dim=None, current_top_k=None, deterministic: bool = False):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        task_emb = self.task_embedding(direction_ids)
        task_emb_seq = jnp.broadcast_to(task_emb[:, None, :], (batch_size, seq_len, task_emb.shape[-1]))
        task_emb_flat = task_emb_seq.reshape(-1, task_emb.shape[-1])

        router_in = jnp.concatenate([x_flat, task_emb_flat], axis=-1)
        router_logits = self.router(router_in.astype(jnp.float32))

        routing_probs = jax.nn.softmax(router_logits, axis=-1)

        # ---------------------------------------------------------
        # DYNAMIC TOP-K WITH MATHEMATICAL MASKING
        # ---------------------------------------------------------
        max_k = self.top_k
        
        # Determine dynamic K for this forward pass
        dyn_k = current_top_k if current_top_k is not None else max_k
        dyn_k = jnp.asarray(dyn_k, dtype=jnp.int32)

        # Always statically extract the top max_k
        top_max_k_probs, top_max_k_indices = jax.lax.top_k(routing_probs, max_k)

        # Create a boolean mask for dyn_k. E.g., if max_k=2 and dyn_k=1 -> [True, False]
        k_mask = jnp.arange(max_k) < dyn_k

        # Mask out probabilities beyond the current dynamic k (this handles k=0 organically)
        masked_probs = jnp.where(k_mask, top_max_k_probs, 0.0)
        
        # Normalize probabilities over the active k experts safely
        prob_sum = jnp.sum(masked_probs, axis=-1, keepdims=True)
        top_k_probs = masked_probs / jnp.maximum(prob_sum, 1e-9)

        # Aux loss computation safely masked
        mask = jax.nn.one_hot(top_max_k_indices, self.num_experts) # (batch*seq, max_k, num_experts)
        active_expert_mask = mask * k_mask[None, :, None]
        
        f_i = jnp.mean(jnp.sum(active_expert_mask, axis=1), axis=0)
        P_i = jnp.mean(routing_probs, axis=0)
        
        dyn_k_float = dyn_k.astype(jnp.float32)
        aux_loss = self.num_experts * jnp.sum(f_i * P_i) * (dyn_k_float / jnp.maximum(self.top_k, 1))

        lse = jax.scipy.special.logsumexp(router_logits, axis=-1)
        z_loss = jnp.mean(jnp.square(lse))

        top_k_probs_cast = top_k_probs.astype(x_flat.dtype)
        routed_output = jnp.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # Select expert only if it was picked AND it's inside the dynamic mask bounds
            expert_mask = (top_max_k_indices == i) & k_mask[None, :]
            expert_outputs = expert(x_flat, current_mlp_dim=current_mlp_dim, deterministic=deterministic)
            routing_weights = jnp.sum(jnp.where(expert_mask, top_k_probs_cast, 0.0), axis=-1)
            routed_output = routed_output + expert_outputs * routing_weights[:, None]

        routed_output = routed_output.reshape(batch_size, seq_len, d_model)

        shared_output = self.shared_expert(x, direction_ids, current_mlp_dim=current_mlp_dim, deterministic=deterministic)

        return routed_output + shared_output, aux_loss, z_loss


class EncoderBlock(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout_rate, cfg.dtype, rngs)
        self.ln_2 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.moe = MoELayer(cfg.d_model, cfg.num_experts, cfg.top_k, cfg.mlp_dim, cfg.dropout_rate, cfg.dtype,
                            cfg.semantic_dim, rngs)
        self.res_dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

    def __call__(self, x, mask, direction_ids, current_mlp_dim=None, current_top_k=None, sin=None, cos=None, deterministic=False):
        attn_out = self.self_attn(self.ln_1(x), mask=mask, sin=sin, cos=cos, deterministic=deterministic)
        x = x + self.res_dropout(attn_out, deterministic=deterministic)

        moe_out, aux_loss, z_loss = self.moe(self.ln_2(x), direction_ids, current_mlp_dim=current_mlp_dim, 
                                             current_top_k=current_top_k, deterministic=deterministic)
        x = x + self.res_dropout(moe_out, deterministic=deterministic)
        return x, aux_loss, z_loss


class DecoderBlock(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout_rate, cfg.dtype, rngs)
        self.ln_2 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout_rate, cfg.dtype, rngs)
        self.ln_3 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.moe = MoELayer(cfg.d_model, cfg.num_experts, cfg.top_k, cfg.mlp_dim, cfg.dropout_rate, cfg.dtype,
                            cfg.semantic_dim, rngs)
        self.res_dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

    def __call__(self, x, tgt_mask, enc_out, src_mask, direction_ids, current_mlp_dim=None, current_top_k=None, sin=None, cos=None, deterministic=False):
        attn_out = self.self_attn(self.ln_1(x), mask=tgt_mask, sin=sin, cos=cos, deterministic=deterministic)
        x = x + self.res_dropout(attn_out, deterministic=deterministic)

        cross_out = self.cross_attn(self.ln_2(x), mask=src_mask, context=enc_out, deterministic=deterministic)
        x = x + self.res_dropout(cross_out, deterministic=deterministic)

        moe_out, aux_loss, z_loss = self.moe(self.ln_3(x), direction_ids, current_mlp_dim=current_mlp_dim,
                                             current_top_k=current_top_k, deterministic=deterministic)
        x = x + self.res_dropout(moe_out, deterministic=deterministic)
        return x, aux_loss, z_loss


class Encoder(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.blocks = [EncoderBlock(cfg, rngs) for _ in range(cfg.num_layers)]
        self.ln_final = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, x, mask, direction_ids, current_mlp_dim=None, current_top_k=None, sin=None, cos=None, deterministic=False):
        total_aux_loss = 0.0
        total_z_loss = 0.0
        for block in self.blocks:
            x, aux_loss, z_loss = block(x, mask, direction_ids, current_mlp_dim=current_mlp_dim, 
                                       current_top_k=current_top_k, sin=sin, cos=cos, deterministic=deterministic)
            total_aux_loss += aux_loss
            total_z_loss += z_loss
        return self.ln_final(x), total_aux_loss, total_z_loss


class Decoder(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.blocks = [DecoderBlock(cfg, rngs) for _ in range(cfg.num_layers)]
        self.ln_final = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, x, tgt_mask, enc_out, src_mask, direction_ids, current_mlp_dim=None, current_top_k=None, sin=None, cos=None, deterministic=False):
        total_aux_loss = 0.0
        total_z_loss = 0.0
        for block in self.blocks:
            x, aux_loss, z_loss = block(x, tgt_mask, enc_out, src_mask, direction_ids, current_mlp_dim=current_mlp_dim, 
                                       current_top_k=current_top_k, sin=sin, cos=cos,
                                        deterministic=deterministic)
            total_aux_loss += aux_loss
            total_z_loss += z_loss
        return self.ln_final(x), total_aux_loss, total_z_loss


class MoETranslationModel(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedding = nnx.Embed(cfg.vocab_size, cfg.d_model, dtype=cfg.dtype, rngs=rngs)

        self.embed_norm = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)

        self.encoder = Encoder(cfg, rngs)
        self.decoder = Decoder(cfg, rngs)

    def _get_rope(self, positions):
        head_dim = self.cfg.d_model // self.cfg.num_heads
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        freqs = jnp.einsum('i,j->ij', positions.astype(jnp.float32), inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        sin = jnp.sin(emb)[None, :, None, :].astype(self.cfg.dtype)
        cos = jnp.cos(emb)[None, :, None, :].astype(self.cfg.dtype)
        return sin, cos

    def encode(self, source_ids, src_mask, current_mlp_dim=None, deterministic=False):
        seq_len = source_ids.shape[1]
        positions = jnp.arange(seq_len)
        direction_ids = (source_ids[:, 0] == self.cfg.vi_en_token_id).astype(jnp.int32)

        sin, cos = self._get_rope(positions)

        x = self.embed_norm(self.embedding(source_ids))
        enc_out, _, _ = self.encoder(x, src_mask, direction_ids, current_mlp_dim=current_mlp_dim, sin=sin, cos=cos, deterministic=deterministic)
        return enc_out

    def decode(self, target_ids, tgt_mask, enc_out, src_mask, current_mlp_dim=None, deterministic=False, direction_ids=None):
        seq_len = target_ids.shape[1]
        positions = jnp.arange(seq_len)
        sin, cos = self._get_rope(positions)

        if direction_ids is None:
            direction_ids = jnp.zeros((target_ids.shape[0],), dtype=jnp.int32)

        x = self.embed_norm(self.embedding(target_ids))
        dec_out, _, _ = self.decoder(x, tgt_mask, enc_out, src_mask, direction_ids, current_mlp_dim=current_mlp_dim, sin=sin, cos=cos,
                                     deterministic=deterministic)

        dec_out_fp32 = dec_out.astype(jnp.float32)
        embed_fp32 = self.embedding.embedding.astype(jnp.float32)

        logits = jnp.dot(dec_out_fp32, embed_fp32.T)
        logits = logits * (self.cfg.d_model ** -0.5)

        cap = 30.0
        logits = cap * jnp.tanh(logits / cap)

        return logits

    def __call__(self, source_ids, target_ids, src_mask, tgt_mask, current_mlp_dim=None, current_top_k=None, deterministic=False):
        seq_len_src = source_ids.shape[1]
        seq_len_tgt = target_ids.shape[1]

        src_positions = jnp.arange(seq_len_src)
        tgt_positions = jnp.arange(seq_len_tgt)

        direction_ids = (source_ids[:, 0] == self.cfg.vi_en_token_id).astype(jnp.int32)

        src_sin, src_cos = self._get_rope(src_positions)
        tgt_sin, tgt_cos = self._get_rope(tgt_positions)

        src_emb = self.embed_norm(self.embedding(source_ids))
        tgt_emb = self.embed_norm(self.embedding(target_ids))

        enc_out, enc_aux, enc_z = self.encoder(src_emb, src_mask, direction_ids, current_mlp_dim=current_mlp_dim, 
                                               current_top_k=current_top_k, sin=src_sin, cos=src_cos,
                                               deterministic=deterministic)
        dec_out, dec_aux, dec_z = self.decoder(tgt_emb, tgt_mask, enc_out, src_mask, direction_ids, current_mlp_dim=current_mlp_dim, 
                                               current_top_k=current_top_k, sin=tgt_sin,
                                               cos=tgt_cos, deterministic=deterministic)

        dec_out_fp32 = dec_out.astype(jnp.float32)
        embed_fp32 = self.embedding.embedding.astype(jnp.float32)

        logits = jnp.dot(dec_out_fp32, embed_fp32.T)
        logits = logits * (self.cfg.d_model ** -0.5)

        cap = 30.0
        logits = cap * jnp.tanh(logits / cap)

        total_moe_layers = self.cfg.num_layers * 2
        avg_aux_loss = (enc_aux + dec_aux) / total_moe_layers
        avg_z_loss = (enc_z + dec_z) / total_moe_layers

        return logits, avg_aux_loss, avg_z_loss

    @property
    def decoder_blocks(self):
        return self.decoder.blocks