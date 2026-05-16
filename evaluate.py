"""
Evaluation script for MoE Translation Model (JAX/Flax NNX).
Evaluates EN->VI and VI->EN on PhoMT test set using BLEU and COMET.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import msgpack
from flax import nnx
from flax.serialization import _msgpack_ext_unpack
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import sacrebleu
import optax
import tensorflow as tf
import orbax.checkpoint as ocp

from config import config as moe_config, MoEModelConfig
from moe_model import MoETranslationModel
from moe_inference import MoEGenerator

tf.config.set_visible_devices([], 'GPU')


def load_moe_model(config, checkpoint_step=None):
    """Load MoE model.
    - checkpoint_step=None  → load latest msgpack (.msg)
    - checkpoint_step="latest" → load latest Orbax checkpoint
    - checkpoint_step=<int> → load specific Orbax checkpoint step
    """
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path_padded)
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vi_en_token_id = tokenizer.convert_tokens_to_ids("<translate-vi-en>")

    model_config = MoEModelConfig(
        vocab_size=vocab_size, pad_token_id=pad_token_id, vi_en_token_id=vi_en_token_id,
        d_model=config.d_model, num_heads=config.num_heads,
        mlp_dim=config.d_ff, num_layers=config.num_layers,
        num_experts=config.num_experts, top_k=config.top_k,
        semantic_dim=config.semantic_dim,
        dropout_rate=0.0, max_seq_len=config.train_max_length_input,
        dtype=jnp.bfloat16
    )

    rngs = nnx.Rngs(config.seed)
    model = MoETranslationModel(model_config, rngs=rngs)
    total_params = sum(x.size for x in jax.tree.leaves(nnx.state(model)))

    if checkpoint_step is not None:
        ckpt_dir = str(config.checkpoint_path.resolve())
        ckpt_manager = ocp.CheckpointManager(ckpt_dir, ocp.StandardCheckpointer())

        if checkpoint_step == "latest":
            step = ckpt_manager.latest_step()
            if step is None:
                print(f"Error: No Orbax checkpoints found in {ckpt_dir}")
                sys.exit(1)
        else:
            step = int(checkpoint_step)

        print(f"Loading Orbax checkpoint step {step} from {ckpt_dir}...")

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.decay_init_value,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            end_value=config.decay_end_value
        )

        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule, weight_decay=0.01))
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        _, abstract_model_state = nnx.split(model)
        _, abstract_opt_state = nnx.split(optimizer)

        restored = ckpt_manager.restore(
            step,
            args=ocp.args.StandardRestore(item={'model': abstract_model_state, 'opt': abstract_opt_state}),
        )
        nnx.update(model, restored['model'])
        del optimizer, abstract_opt_state
        print(f"Orbax checkpoint step {step} loaded.")
        return model, tokenizer, total_params, step

    # Default: load latest msgpack
    if config.latest_msg_path.exists():
        print(f"Loading msgpack weights from {config.latest_msg_path}...")
        with open(config.latest_msg_path, "rb") as f:
            msgpack_bytes = f.read()

        saved_dict = msgpack.unpackb(msgpack_bytes, ext_hook=_msgpack_ext_unpack, raw=False, strict_map_key=False)
        _, current_params, _ = nnx.split(model, nnx.Param, ...)

        def wrap_state(template, raw_dict):
            if hasattr(template, "items"):
                res = {}
                for k, v in template.items():
                    val = raw_dict.get(k)
                    if val is None and isinstance(k, str) and k.isdigit():
                        val = raw_dict.get(int(k))
                    if val is None and isinstance(k, int):
                        val = raw_dict.get(str(k))
                    res[k] = wrap_state(v, val)
                return nnx.State(res)
            elif isinstance(template, nnx.Variable):
                return type(template)(raw_dict)
            return raw_dict

        restored_params = wrap_state(current_params, saved_dict)
        nnx.update(model, restored_params)
        print("Msgpack weights loaded.")
        return model, tokenizer, total_params, "latest_msg"

    print(f"Error: No msgpack at {config.latest_msg_path}")
    sys.exit(1)


def load_comet_model():
    try:
        from comet import download_model, load_from_checkpoint
        import torch
        print("Loading COMET model (Unbabel/wmt22-comet-da)...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda:0"))
        model.eval()
        print("COMET model loaded.")
        return model
    except Exception as e:
        print(f"WARNING: Could not load COMET model: {e}")
        return None


def compute_comet(comet_model, sources, predictions, references, batch_size=64):
    if comet_model is None:
        return None
    import torch
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
    print(f"Computing COMET scores ({len(data)} samples)...")
    with torch.no_grad():
        output = comet_model.predict(
            data, batch_size=batch_size,
            gpus=1 if torch.cuda.is_available() else 0,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            progress_bar=True
        )
    score = float(output.system_score)
    print(f"  COMET: {score:.4f}")
    return score


def clean_detokenize(texts):
    """Fix punctuation spacing artifacts from tokenizer output.
    Matches the same post-processing used during training BLEU evaluation."""
    cleaned = []
    for t in texts:
        t = t.replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
        t = t.replace(" !", "!").replace(" :", ":").replace(" ;", ";")
        cleaned.append(t.strip())
    return cleaned


def run_inference(generator, sources, src_lang, tgt_lang, batch_size, max_length,
                  num_beams=1, length_penalty=0.6):
    prefix = f"<translate-{src_lang}-{tgt_lang}> "
    method = "beam" if num_beams > 1 else "greedy"
    predictions = []
    total_tokens = 0

    start_time = time.time()
    for i in tqdm(range(0, len(sources), batch_size), desc=f"Inference {src_lang}->{tgt_lang}"):
        batch_src = sources[i:i+batch_size]
        inputs = [prefix + str(s) for s in batch_src]
        decoded, metrics = generator.generate(
            inputs, method=method, max_len=max_length,
            num_beams=num_beams, verbose=False)
        predictions.extend(decoded)
        total_tokens += metrics['tokens_out']

    elapsed = time.time() - start_time
    print(f"Inference: {elapsed:.2f}s | {len(sources)/elapsed:.1f} samples/s | {total_tokens/elapsed:.1f} tokens/s")
    return predictions, elapsed, total_tokens


def evaluate_direction(generator, dataset, src_lang, tgt_lang, batch_size, max_length,
                       limit=None, num_beams=1, length_penalty=0.6):
    print(f"\n{'='*70}")
    print(f"  {src_lang.upper()} -> {tgt_lang.upper()}")
    print('='*70)

    if src_lang in dataset.column_names:
        src_field = src_lang
    else:
        src_field = f"original_{src_lang}"
    if tgt_lang in dataset.column_names:
        tgt_field = tgt_lang
    else:
        tgt_field = f"original_{tgt_lang}"

    filtered = dataset.filter(
        lambda x: x[src_field] is not None and x[tgt_field] is not None
                  and len(str(x[src_field]).strip()) > 0
                  and len(str(x[tgt_field]).strip()) > 0,
        num_proc=16
    )

    sources = list(filtered[src_field])
    references = list(filtered[tgt_field])

    if limit and limit < len(sources):
        sources = sources[:limit]
        references = references[:limit]

    print(f"Samples: {len(sources)}")

    predictions, inference_time, total_tokens = run_inference(
        generator, sources, src_lang, tgt_lang, batch_size, max_length,
        num_beams=num_beams
    )

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    score = float(bleu.score)

    tps = total_tokens / inference_time if inference_time > 0 else 0
    print(f"\n  BLEU: {score:.2f}  |  {bleu}")
    print(f"  Time: {inference_time:.2f}s  |  {tps:.1f} tokens/s")

    print(f"\n  Examples:")
    for i in range(min(5, len(predictions))):
        print(f"    SRC: {sources[i]}")
        print(f"    REF: {references[i]}")
        print(f"    HYP: {predictions[i]}")
        print()

    return score, predictions, references, sources


def main():
    parser = argparse.ArgumentParser(description='Evaluate MoE Translation Model')
    parser.add_argument('--dataset-name', type=str, default='ura-hcmut/PhoMT')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--length-penalty', type=float, default=0.6)
    parser.add_argument('--output-dir', type=str, default='outputs/moe_eval')
    parser.add_argument('--checkpoint-step', type=str, default=None,
                        help='Orbax checkpoint step to load. "latest" for latest checkpoint, '
                             'or a number (e.g. 160000). Default: load .msg file')
    parser.add_argument('--skip-comet', action='store_true',
                        help='Skip COMET evaluation (only report BLEU)')
    args = parser.parse_args()

    MAX_GEN_LENGTH = args.max_length if args.max_length else moe_config.max_length_inference
    MAX_GEN_LENGTH = min(MAX_GEN_LENGTH, moe_config.max_length_inference)
    MAX_INPUT_LENGTH = moe_config.train_max_length_input
    NUM_BEAMS = args.num_beams
    LENGTH_PENALTY = args.length_penalty

    decode_method = f"beam(b={NUM_BEAMS})" if NUM_BEAMS > 1 else "greedy"

    print("\n" + "="*70)
    print("  MoE Translation Model Evaluation")
    print("="*70)
    print(f"  JAX devices:  {jax.devices()}")
    print(f"  Config:       {moe_config.signature} v{moe_config.version}")
    print(f"  Dataset:      {args.dataset_name} ({args.split})")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Max length:   {MAX_GEN_LENGTH}")
    print(f"  Decoding:     {decode_method}")
    print(f"  Checkpoint:   {args.checkpoint_step or 'latest .msg'}")
    print(f"  COMET:        {'skip' if args.skip_comet else 'wmt22-comet-da'}")
    print("="*70 + "\n")

    model, tokenizer, total_params, ckpt_label = load_moe_model(moe_config, args.checkpoint_step)
    generator = MoEGenerator(model, tokenizer, max_input_len=MAX_INPUT_LENGTH)

    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    print("Warming up JIT...")
    _ = generator.generate(["Warmup text"], method="greedy", max_len=MAX_GEN_LENGTH, verbose=False)
    print("JIT warmup complete.\n")

    print(f"Loading dataset: {args.dataset_name}")
    if os.path.exists(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)

    from datasets import DatasetDict
    if isinstance(dataset, DatasetDict):
        if args.split in dataset:
            test_data = dataset[args.split]
        elif "test" in dataset:
            test_data = dataset["test"]
        else:
            test_data = dataset[list(dataset.keys())[0]]
    else:
        test_data = dataset

    print(f"Dataset loaded: {len(test_data)} examples\n")

    bleu_en_vi, preds_en_vi, refs_en_vi, srcs_en_vi = evaluate_direction(
        generator, test_data, "en", "vi", args.batch_size, MAX_GEN_LENGTH,
        args.limit, num_beams=NUM_BEAMS, length_penalty=LENGTH_PENALTY
    )

    bleu_vi_en, preds_vi_en, refs_vi_en, srcs_vi_en = evaluate_direction(
        generator, test_data, "vi", "en", args.batch_size, MAX_GEN_LENGTH,
        args.limit, num_beams=NUM_BEAMS, length_penalty=LENGTH_PENALTY
    )

    avg_bleu = (bleu_en_vi + bleu_vi_en) / 2

    comet_en_vi = None
    comet_vi_en = None
    avg_comet = None
    if not args.skip_comet:
        comet_model = load_comet_model()
        if comet_model is not None:
            print("\nComputing COMET scores...")
            print(f"\n  EN -> VI:")
            comet_en_vi = compute_comet(comet_model, srcs_en_vi, preds_en_vi, refs_en_vi)
            print(f"\n  VI -> EN:")
            comet_vi_en = compute_comet(comet_model, srcs_vi_en, preds_vi_en, refs_vi_en)
            if comet_en_vi is not None and comet_vi_en is not None:
                avg_comet = (comet_en_vi + comet_vi_en) / 2

    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"  Checkpoint:   {ckpt_label}")
    print(f"  Parameters:   {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Decoding:     {decode_method}")
    print(f"  en->vi BLEU:  {bleu_en_vi:.2f}")
    print(f"  vi->en BLEU:  {bleu_vi_en:.2f}")
    print(f"  Average BLEU: {avg_bleu:.2f}")
    if comet_en_vi is not None:
        print(f"  en->vi COMET: {comet_en_vi:.4f}")
    if comet_vi_en is not None:
        print(f"  vi->en COMET: {comet_vi_en:.4f}")
    if avg_comet is not None:
        print(f"  Average COMET:{avg_comet:.4f}")
    print("="*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(ckpt_label),
        "config": f"{moe_config.signature} v{moe_config.version}",
        "params": total_params,
        "decoding": decode_method,
        "bleu": {
            "en->vi": bleu_en_vi,
            "vi->en": bleu_vi_en,
            "average": avg_bleu,
        },
    }
    if comet_en_vi is not None:
        results["comet"] = {
            "model": "Unbabel/wmt22-comet-da",
            "en->vi": comet_en_vi,
            "vi->en": comet_vi_en,
            "average": avg_comet,
        }
    results_file = output_dir / f"moe_eval_results_{ckpt_label}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    for direction, preds, refs, srcs in [
        ("en_vi", preds_en_vi, refs_en_vi, srcs_en_vi),
        ("vi_en", preds_vi_en, refs_vi_en, srcs_vi_en)
    ]:
        pred_file = output_dir / f"predictions_{direction}_{ckpt_label}.txt"
        with open(pred_file, 'w', encoding='utf-8') as f:
            for src, pred, ref in zip(srcs, preds, refs):
                f.write(f"SRC: {src}\nREF: {ref}\nHYP: {pred}\n" + "-"*70 + "\n")
        print(f"Predictions saved to: {pred_file}")


if __name__ == "__main__":
    main()
