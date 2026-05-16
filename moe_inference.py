import time
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


@nnx.jit(static_argnames=('max_len', 'pad_id', 'eos_id', 'current_mlp_dim'))
def generate_fast_greedy_jitted(model, source_ids, src_mask, max_len, pad_id, eos_id, current_mlp_dim=None):
    batch_size = source_ids.shape[0]
    num_layers = model.cfg.num_layers
    num_heads = model.cfg.num_heads
    head_dim = model.cfg.d_model // num_heads

    direction_ids = (source_ids[:, 0] == model.cfg.vi_en_token_id).astype(jnp.int32)

    enc_out = model.encode(source_ids, src_mask, current_mlp_dim=current_mlp_dim, deterministic=True)
    cross_mask = (src_mask[:, None, None, :] == 1)

    k_cache = jnp.zeros((num_layers, batch_size, max_len, num_heads, head_dim), dtype=model.cfg.dtype)
    v_cache = jnp.zeros((num_layers, batch_size, max_len, num_heads, head_dim), dtype=model.cfg.dtype)

    cross_k_cache, cross_v_cache = [], []
    for block in model.decoder_blocks:
        ck_raw = block.cross_attn.key(enc_out).reshape(batch_size, -1, num_heads, head_dim)
        ck = block.cross_attn.k_norm(ck_raw)
        cv = block.cross_attn.value(enc_out).reshape(batch_size, -1, num_heads, head_dim)
        cross_k_cache.append(ck)
        cross_v_cache.append(cv)

    target_ids = jnp.full((batch_size, max_len), pad_id, dtype=jnp.int32)
    is_finished = jnp.zeros((batch_size, 1), dtype=jnp.bool_)

    def cond_body(val):
        i, _, is_finished, _, _ = val
        not_all_finished = jnp.logical_not(jnp.all(is_finished))
        return jnp.logical_and(i < max_len, not_all_finished)

    def loop_body(val):
        i, target_ids, is_finished, k_cache, v_cache = val

        x_t_id = jax.lax.dynamic_slice_in_dim(target_ids, i - 1, 1, axis=1)
        x_t_id = jnp.where(i == 1, pad_id, x_t_id)

        dec_x = model.embed_norm(model.embedding(x_t_id))

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
        freqs = (i - 1) * inv_freq
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        sin = jnp.sin(emb)[None, None, None, :].astype(model.cfg.dtype)
        cos = jnp.cos(emb)[None, None, None, :].astype(model.cfg.dtype)

        for layer_idx, block in enumerate(model.decoder_blocks):
            x_norm = block.ln_1(dec_x)
            q = block.self_attn.query(x_norm).reshape(batch_size, 1, num_heads, head_dim)
            k = block.self_attn.key(x_norm).reshape(batch_size, 1, num_heads, head_dim)
            v = block.self_attn.value(x_norm).reshape(batch_size, 1, num_heads, head_dim)

            q = block.self_attn.q_norm(q)
            k = block.self_attn.k_norm(k)

            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            q = q.astype(model.cfg.dtype)
            k = k.astype(model.cfg.dtype)
            v = v.astype(model.cfg.dtype)

            k_update = k[None, ...]
            v_update = v[None, ...]
            k_cache = jax.lax.dynamic_update_slice(k_cache, k_update, (layer_idx, 0, i - 1, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(v_cache, v_update, (layer_idx, 0, i - 1, 0, 0))

            k_layer = k_cache[layer_idx]
            v_layer = v_cache[layer_idx]

            attn = jnp.einsum('bqhd,bkhd->bhqk', q, k_layer) / jnp.sqrt(head_dim)
            idx = jnp.arange(max_len)[None, None, None, :]
            attn = jnp.where(idx <= (i - 1), attn, -jnp.inf)
            attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)

            attn_out = jnp.einsum('bhqk,bkhd->bqhd', attn, v_layer).reshape(batch_size, 1, -1)
            dec_x = dec_x + block.self_attn.out(attn_out)

            x_norm2 = block.ln_2(dec_x)
            q_cross = block.cross_attn.query(x_norm2).reshape(batch_size, 1, num_heads, head_dim)
            q_cross = block.cross_attn.q_norm(q_cross)
            ck, cv = cross_k_cache[layer_idx], cross_v_cache[layer_idx]

            attn_cross = jnp.einsum('bqhd,bkhd->bhqk', q_cross, ck) / jnp.sqrt(head_dim)
            attn_cross = jnp.where(cross_mask, attn_cross, -jnp.inf)
            attn_cross = jax.nn.softmax(attn_cross.astype(jnp.float32), axis=-1).astype(q_cross.dtype)

            cross_out = jnp.einsum('bhqk,bkhd->bqhd', attn_cross, cv).reshape(batch_size, 1, -1)
            dec_x = dec_x + block.cross_attn.out(cross_out)

            moe_out, _, _ = block.moe(block.ln_3(dec_x), direction_ids, current_mlp_dim=current_mlp_dim, deterministic=True)
            dec_x = dec_x + moe_out

        dec_x = model.decoder.ln_final(dec_x)

        dec_x_fp32 = dec_x.astype(jnp.float32)
        embed_fp32 = model.embedding.embedding.astype(jnp.float32)
        logits = jnp.dot(dec_x_fp32, embed_fp32.T)
        logits = logits * (model.cfg.d_model ** -0.5)

        cap = 30.0
        logits = cap * jnp.tanh(logits / cap)

        next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)

        next_token = jnp.where(is_finished, pad_id, next_token)
        is_finished = is_finished | (next_token == eos_id)
        target_ids = jax.lax.dynamic_update_slice(target_ids, next_token, (0, i))

        return (i + 1, target_ids, is_finished, k_cache, v_cache)

    initial_val = (1, target_ids, is_finished, k_cache, v_cache)
    final_val = jax.lax.while_loop(cond_body, loop_body, initial_val)
    return final_val[1], final_val[0]


@nnx.jit(static_argnames=('max_len', 'top_k', 'pad_id', 'eos_id', 'current_mlp_dim'))
def generate_fast_sample_jitted(model, source_ids, src_mask, key, max_len, temperature, top_k, top_p, pad_id, eos_id, current_mlp_dim=None):
    batch_size = source_ids.shape[0]
    num_layers = model.cfg.num_layers
    num_heads = model.cfg.num_heads
    head_dim = model.cfg.d_model // num_heads

    direction_ids = (source_ids[:, 0] == model.cfg.vi_en_token_id).astype(jnp.int32)

    enc_out = model.encode(source_ids, src_mask, current_mlp_dim=current_mlp_dim, deterministic=True)
    cross_mask = (src_mask[:, None, None, :] == 1)

    k_cache = jnp.zeros((num_layers, batch_size, max_len, num_heads, head_dim), dtype=model.cfg.dtype)
    v_cache = jnp.zeros((num_layers, batch_size, max_len, num_heads, head_dim), dtype=model.cfg.dtype)

    cross_k_cache, cross_v_cache = [], []
    for block in model.decoder_blocks:
        ck_raw = block.cross_attn.key(enc_out).reshape(batch_size, -1, num_heads, head_dim)
        ck = block.cross_attn.k_norm(ck_raw)
        cv = block.cross_attn.value(enc_out).reshape(batch_size, -1, num_heads, head_dim)
        cross_k_cache.append(ck)
        cross_v_cache.append(cv)

    target_ids = jnp.full((batch_size, max_len), pad_id, dtype=jnp.int32)
    is_finished = jnp.zeros((batch_size, 1), dtype=jnp.bool_)

    def cond_body(val):
        i, _, is_finished, _, _, _ = val
        not_all_finished = jnp.logical_not(jnp.all(is_finished))
        return jnp.logical_and(i < max_len, not_all_finished)

    def loop_body(val):
        i, target_ids, is_finished, k_cache, v_cache, rng_key = val
        rng_key, subkey = jax.random.split(rng_key)

        x_t_id = jax.lax.dynamic_slice_in_dim(target_ids, i - 1, 1, axis=1)
        x_t_id = jnp.where(i == 1, pad_id, x_t_id)

        dec_x = model.embed_norm(model.embedding(x_t_id))

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
        freqs = (i - 1) * inv_freq
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        sin = jnp.sin(emb)[None, None, None, :].astype(model.cfg.dtype)
        cos = jnp.cos(emb)[None, None, None, :].astype(model.cfg.dtype)

        for layer_idx, block in enumerate(model.decoder_blocks):
            x_norm = block.ln_1(dec_x)
            q = block.self_attn.query(x_norm).reshape(batch_size, 1, num_heads, head_dim)
            k = block.self_attn.key(x_norm).reshape(batch_size, 1, num_heads, head_dim)
            v = block.self_attn.value(x_norm).reshape(batch_size, 1, num_heads, head_dim)

            q = block.self_attn.q_norm(q)
            k = block.self_attn.k_norm(k)

            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            q = q.astype(model.cfg.dtype)
            k = k.astype(model.cfg.dtype)
            v = v.astype(model.cfg.dtype)

            k_update = k[None, ...]
            v_update = v[None, ...]
            k_cache = jax.lax.dynamic_update_slice(k_cache, k_update, (layer_idx, 0, i - 1, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(v_cache, v_update, (layer_idx, 0, i - 1, 0, 0))

            k_layer = k_cache[layer_idx]
            v_layer = v_cache[layer_idx]

            attn = jnp.einsum('bqhd,bkhd->bhqk', q, k_layer) / jnp.sqrt(head_dim)
            idx = jnp.arange(max_len)[None, None, None, :]
            attn = jnp.where(idx <= (i - 1), attn, -jnp.inf)
            attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)

            attn_out = jnp.einsum('bhqk,bkhd->bqhd', attn, v_layer).reshape(batch_size, 1, -1)
            dec_x = dec_x + block.self_attn.out(attn_out)

            x_norm2 = block.ln_2(dec_x)
            q_cross = block.cross_attn.query(x_norm2).reshape(batch_size, 1, num_heads, head_dim)
            q_cross = block.cross_attn.q_norm(q_cross)
            ck, cv = cross_k_cache[layer_idx], cross_v_cache[layer_idx]

            attn_cross = jnp.einsum('bqhd,bkhd->bhqk', q_cross, ck) / jnp.sqrt(head_dim)
            attn_cross = jnp.where(cross_mask, attn_cross, -jnp.inf)
            attn_cross = jax.nn.softmax(attn_cross.astype(jnp.float32), axis=-1).astype(q_cross.dtype)

            cross_out = jnp.einsum('bhqk,bkhd->bqhd', attn_cross, cv).reshape(batch_size, 1, -1)
            dec_x = dec_x + block.cross_attn.out(cross_out)

            moe_out, _, _ = block.moe(block.ln_3(dec_x), direction_ids, current_mlp_dim=current_mlp_dim, 
                                     deterministic=True)
            dec_x = dec_x + moe_out

        dec_x = model.decoder.ln_final(dec_x)

        dec_x_fp32 = dec_x.astype(jnp.float32)
        embed_fp32 = model.embedding.embedding.astype(jnp.float32)
        logits = jnp.dot(dec_x_fp32, embed_fp32.T)
        logits = logits * (model.cfg.d_model ** -0.5)

        cap = 30.0
        logits = cap * jnp.tanh(logits / cap)

        next_logits = logits[:, 0, :] / temperature

        top_k_logits, _ = jax.lax.top_k(next_logits, top_k)
        next_logits = jnp.where(next_logits < top_k_logits[:, -1:], -jnp.inf, next_logits)

        probs = jax.nn.softmax(next_logits, axis=-1)
        sorted_probs = jnp.sort(probs, axis=-1)[:, ::-1]
        cumsum = jnp.cumsum(sorted_probs, axis=-1)

        mask = cumsum < top_p
        mask = jnp.concatenate([jnp.ones_like(mask[:, :1]), mask[:, :-1]], axis=-1)
        threshold_prob = jnp.min(jnp.where(mask, sorted_probs, 1.0), axis=-1, keepdims=True)

        next_logits = jnp.where(probs >= threshold_prob, next_logits, -jnp.inf)
        next_token = jax.random.categorical(subkey, next_logits, axis=-1)[:, None].astype(jnp.int32)

        next_token = jnp.where(is_finished, pad_id, next_token)
        is_finished = is_finished | (next_token == eos_id)
        target_ids = jax.lax.dynamic_update_slice(target_ids, next_token, (0, i))

        return (i + 1, target_ids, is_finished, k_cache, v_cache, rng_key)

    initial_val = (1, target_ids, is_finished, k_cache, v_cache, key)
    final_val = jax.lax.while_loop(cond_body, loop_body, initial_val)
    return final_val[1], final_val[0]


@nnx.jit(static_argnames=('max_len', 'pad_id', 'eos_id', 'beam_size', 'current_mlp_dim'))
def generate_fast_beam_jitted(model, source_ids, src_mask, max_len, pad_id, eos_id, beam_size=4, current_mlp_dim=None):
    length_penalty = 0.6
    batch_size = source_ids.shape[0]
    num_layers = model.cfg.num_layers
    num_heads = model.cfg.num_heads
    head_dim = model.cfg.d_model // num_heads

    direction_ids = (source_ids[:, 0] == model.cfg.vi_en_token_id).astype(jnp.int32)
    enc_out = model.encode(source_ids, src_mask, current_mlp_dim=current_mlp_dim, deterministic=True)
    cross_mask = (src_mask[:, None, None, :] == 1)

    enc_out = jnp.repeat(enc_out, beam_size, axis=0)
    cross_mask = jnp.repeat(cross_mask, beam_size, axis=0)
    direction_ids = jnp.repeat(direction_ids, beam_size, axis=0)

    k_cache = jnp.zeros((num_layers, batch_size * beam_size, max_len, num_heads, head_dim), dtype=model.cfg.dtype)
    v_cache = jnp.zeros((num_layers, batch_size * beam_size, max_len, num_heads, head_dim), dtype=model.cfg.dtype)

    cross_k_cache, cross_v_cache = [], []
    for block in model.decoder_blocks:
        ck_raw = block.cross_attn.key(enc_out).reshape(batch_size * beam_size, -1, num_heads, head_dim)
        ck = block.cross_attn.k_norm(ck_raw)
        cv = block.cross_attn.value(enc_out).reshape(batch_size * beam_size, -1, num_heads, head_dim)
        cross_k_cache.append(ck)
        cross_v_cache.append(cv)

    target_ids = jnp.full((batch_size, beam_size, max_len), pad_id, dtype=jnp.int32)
    is_finished = jnp.zeros((batch_size, beam_size), dtype=jnp.bool_)
    lengths = jnp.ones((batch_size, beam_size), dtype=jnp.int32)

    scores = jnp.full((batch_size, beam_size), -1e9)
    scores = scores.at[:, 0].set(0.0)

    def cond_body(val):
        i, _, is_finished, _, _, _, _ = val
        not_all_finished = jnp.logical_not(jnp.all(is_finished))
        return jnp.logical_and(i < max_len, not_all_finished)

    def loop_body(val):
        i, target_ids, is_finished, scores, lengths, k_cache, v_cache = val

        x_t_id = jax.lax.dynamic_slice_in_dim(target_ids, i - 1, 1, axis=2)
        x_t_id = x_t_id.reshape(batch_size * beam_size, 1)
        x_t_id = jnp.where(i == 1, pad_id, x_t_id)

        dec_x = model.embed_norm(model.embedding(x_t_id))
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
        freqs = (i - 1) * inv_freq
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        sin = jnp.sin(emb)[None, None, None, :].astype(model.cfg.dtype)
        cos = jnp.cos(emb)[None, None, None, :].astype(model.cfg.dtype)

        for layer_idx, block in enumerate(model.decoder_blocks):
            x_norm = block.ln_1(dec_x)
            q = block.self_attn.query(x_norm).reshape(batch_size * beam_size, 1, num_heads, head_dim)
            k = block.self_attn.key(x_norm).reshape(batch_size * beam_size, 1, num_heads, head_dim)
            v = block.self_attn.value(x_norm).reshape(batch_size * beam_size, 1, num_heads, head_dim)

            q = block.self_attn.q_norm(q)
            k = block.self_attn.k_norm(k)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            q = q.astype(model.cfg.dtype)
            k = k.astype(model.cfg.dtype)
            v = v.astype(model.cfg.dtype)

            k_cache = jax.lax.dynamic_update_slice(k_cache, k[None, ...], (layer_idx, 0, i - 1, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(v_cache, v[None, ...], (layer_idx, 0, i - 1, 0, 0))

            attn = jnp.einsum('bqhd,bkhd->bhqk', q, k_cache[layer_idx]) / jnp.sqrt(head_dim)
            idx = jnp.arange(max_len)[None, None, None, :]
            attn = jnp.where(idx <= (i - 1), attn, -jnp.inf)
            attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)

            attn_out = jnp.einsum('bhqk,bkhd->bqhd', attn, v_cache[layer_idx]).reshape(batch_size * beam_size, 1, -1)
            dec_x = dec_x + block.self_attn.out(attn_out)

            x_norm2 = block.ln_2(dec_x)
            q_cross = block.cross_attn.query(x_norm2).reshape(batch_size * beam_size, 1, num_heads, head_dim)
            q_cross = block.cross_attn.q_norm(q_cross)
            ck, cv = cross_k_cache[layer_idx], cross_v_cache[layer_idx]

            attn_cross = jnp.einsum('bqhd,bkhd->bhqk', q_cross, ck) / jnp.sqrt(head_dim)
            attn_cross = jnp.where(cross_mask, attn_cross, -jnp.inf)
            attn_cross = jax.nn.softmax(attn_cross.astype(jnp.float32), axis=-1).astype(q_cross.dtype)

            cross_out = jnp.einsum('bhqk,bkhd->bqhd', attn_cross, cv).reshape(batch_size * beam_size, 1, -1)
            dec_x = dec_x + block.cross_attn.out(cross_out)

            moe_out, _, _ = block.moe(block.ln_3(dec_x), direction_ids, current_mlp_dim=current_mlp_dim,
                                     deterministic=True)
            dec_x = dec_x + moe_out

        dec_x = model.decoder.ln_final(dec_x)
        dec_x_fp32 = dec_x.astype(jnp.float32)
        embed_fp32 = model.embedding.embedding.astype(jnp.float32)
        logits = jnp.dot(dec_x_fp32, embed_fp32.T)
        logits = logits * (model.cfg.d_model ** -0.5)
        logits = 30.0 * jnp.tanh(logits / 30.0)

        logits = logits.reshape(batch_size, beam_size, model.cfg.vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        pad_mask = jnp.where(jnp.arange(model.cfg.vocab_size) == pad_id, 0.0, -1e9)
        log_probs = jnp.where(is_finished[..., None], pad_mask, log_probs)

        next_scores = scores[..., None] + log_probs
        next_scores_flat = next_scores.reshape(batch_size, -1)

        topk_scores, topk_indices = jax.lax.top_k(next_scores_flat, beam_size)

        beam_idx = topk_indices // model.cfg.vocab_size
        token_idx = topk_indices % model.cfg.vocab_size
        batch_idx = jnp.arange(batch_size)[:, None]

        gathered_target_ids = target_ids[batch_idx, beam_idx, :]
        target_ids = jax.lax.dynamic_update_slice(gathered_target_ids, token_idx[..., None], (0, 0, i))

        is_finished = is_finished[batch_idx, beam_idx]
        lengths = lengths[batch_idx, beam_idx]

        new_finished = is_finished | (token_idx == eos_id)
        lengths = jnp.where(is_finished, lengths, lengths + 1)

        def gather_cache(cache):
            c = cache.reshape(num_layers, batch_size, beam_size, max_len, num_heads, head_dim)
            c = c[jnp.arange(num_layers)[:, None, None], batch_idx, beam_idx, :, :, :]
            return c.reshape(num_layers, batch_size * beam_size, max_len, num_heads, head_dim)

        k_cache = gather_cache(k_cache)
        v_cache = gather_cache(v_cache)

        return i + 1, target_ids, new_finished, topk_scores, lengths, k_cache, v_cache

    initial_val = (1, target_ids, is_finished, scores, lengths, k_cache, v_cache)
    final_val = jax.lax.while_loop(cond_body, loop_body, initial_val)

    final_target_ids = final_val[1]
    final_scores = final_val[3]
    final_lengths = final_val[4]

    final_scores = final_scores / (final_lengths ** length_penalty)

    best_beam_idx = jnp.argmax(final_scores, axis=-1)
    batch_idx = jnp.arange(batch_size)
    best_target_ids = final_target_ids[batch_idx, best_beam_idx, :]

    return best_target_ids, final_val[0]


class MoEGenerator:
    def __init__(self, model, tokenizer, max_input_len=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

        self.C_BLUE = '\033[94m'
        self.C_GREEN = '\033[92m'
        self.C_YELLOW = '\033[93m'
        self.C_END = '\033[0m'
        self.C_BOLD = '\033[1m'

    def generate(self, texts, method="greedy", max_len=60, temperature=0.7, top_k=40, top_p=0.9, seed=42,
                 num_beams=4, length_penalty=0.6, current_mlp_dim=None, verbose=True):
        inputs = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=self.max_input_len, return_tensors="np"
        )
        src_ids = jnp.array(inputs.input_ids.astype(jnp.int32))
        src_mask = jnp.array(inputs.attention_mask.astype(jnp.int32))
        tokens_in = int(np.sum(inputs.attention_mask))
        batch_size = len(texts)

        t0 = time.time()

        if method == "greedy":
            out_ids, final_i = generate_fast_greedy_jitted(self.model, src_ids, src_mask, max_len, self.pad_id,
                                                           self.eos_id, current_mlp_dim=current_mlp_dim)
        elif method == "beam":
            out_ids, final_i = generate_fast_beam_jitted(self.model, src_ids, src_mask, max_len, self.pad_id,
                                                         self.eos_id, beam_size=num_beams, current_mlp_dim=current_mlp_dim)
        elif method == "sample":
            key = jax.random.key(seed)
            out_ids, final_i = generate_fast_sample_jitted(self.model, src_ids, src_mask, key, max_len, temperature,
                                                           top_k, top_p, self.pad_id, self.eos_id, current_mlp_dim=current_mlp_dim)
        else:
            raise ValueError("Method must be 'greedy', 'beam', or 'sample'")

        out_ids = out_ids.block_until_ready()
        run_time = max(time.time() - t0, 1e-9)

        out_ids_np = np.array(out_ids)
        hardware_steps = int(np.array(final_i)) - 1

        tokens_out = int(np.sum((out_ids_np != self.pad_id) & (out_ids_np != self.eos_id)))
        decoded_texts = self.tokenizer.batch_decode(out_ids_np, skip_special_tokens=True)

        tps_effective = tokens_out / run_time
        tps_hardware = (hardware_steps * batch_size) / run_time

        metrics = {
            "method": method, "batch_size": batch_size, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "run_time_s": run_time, "tps_effective": tps_effective,
            "tps_hardware": tps_hardware, "hardware_steps": hardware_steps
        }

        if verbose:
            color = self.C_BLUE if method == "greedy" else self.C_YELLOW
            title = f" [{method.upper()} GENERATE] "
            print(f"\n{self.C_BOLD}{color}┌{title.ljust(64, '─')}{self.C_END}")
            print(
                f"{color}│{self.C_END} Batch: {batch_size:<5} | Tokens In: {tokens_in:<5} | Tokens Out: {tokens_out:<5}")
            print(
                f"{color}│{self.C_END} Time:  {run_time:<5.3f}s | {self.C_BOLD}Speed: {self.C_GREEN}{tps_effective:,.1f} TPS{self.C_END} (Effective)")
            print(f"{self.C_BOLD}{color}└{'─' * 64}{self.C_END}")

        return decoded_texts, metrics