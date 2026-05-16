import sys
import os
import glob
from functools import partial

# Prevent Rust Tokenizer deadlocks in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.extend(['/mnt/data/edw_2'])
import tensorflow as tf

# Disable GPU to avoid consuming memory JAX will need later
tf.config.set_visible_devices([], 'GPU')

from datasets import load_from_disk, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from config import config

direction_map = {
    "<translate-en-vi>": ("en", "vi"),
    "<translate-vi-en>": ("vi", "en"),
}

def map_and_pack_translation(batch, direction, max_length, tokenizer):
    source_lang, target_lang = direction_map[direction]
    
    packed_inputs = []
    packed_targets = []
    
    cur_input = ""
    cur_target = ""
    
    delimiter = "\n"
    
    # We extract the languages properly
    # Some datasets might have 'translation' feature, some might have flat 'en' and 'vi'
    if 'translation' in batch:
        src_texts = [ex[source_lang] for ex in batch['translation']]
        tgt_texts = [ex[target_lang] for ex in batch['translation']]
    else:
        src_texts = batch[source_lang]
        tgt_texts = batch[target_lang]
    
    for i in range(len(src_texts)):
        src_text = str(src_texts[i]).strip()
        tgt_text = str(tgt_texts[i]).strip()
        
        if not src_text or not tgt_text:
            continue
            
        cand_input = f"{cur_input}{delimiter}{src_text}" if cur_input else src_text
        cand_target = f"{cur_target}{delimiter}{tgt_text}" if cur_target else tgt_text
        
        # Rough estimation: 1 char ~ 0.25 tokens. max_length * 3 is safe.
        if len(cand_input) < max_length * 3.5 and len(cand_target) < max_length * 3.5:
            cur_input = cand_input
            cur_target = cand_target
        else:
            if cur_input:
                packed_inputs.append(f"{direction} {cur_input}")
                packed_targets.append(cur_target)
            cur_input = src_text
            cur_target = tgt_text
            
    if cur_input:
        packed_inputs.append(f"{direction} {cur_input}")
        packed_targets.append(cur_target)
        
    model_inputs = tokenizer(
        packed_inputs, max_length=max_length, truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        packed_targets, max_length=max_length, truncation=True,
        padding="max_length"
    ).input_ids
    
    model_inputs["labels"] = labels
    return model_inputs

def cast_dtypes(features):
    features['input_ids'] = tf.cast(features['input_ids'], tf.int32)
    features['attention_mask'] = tf.cast(features['attention_mask'], tf.int32)
    features['labels'] = tf.cast(features['labels'], tf.int32)
    return features

def find_hf_datasets(base_dir):
    paths = []
    for root, dirs, files in os.walk(base_dir):
        if 'dataset_info.json' in files:
            paths.append(root)
    return paths

def main():
    print(f'Python {sys.version} on {sys.platform}')

    # 1. Prepare Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path_padded)
    if "\n" not in tokenizer.get_vocab():
        print("Adding '\\n' to tokenizer...")
        tokenizer.add_tokens(["\n"])
        # MatMoE requires vocab size to be multiple of 8
        vocab_size = len(tokenizer)
        if vocab_size % 8 != 0:
            pad_size = 8 - (vocab_size % 8)
            pad_tokens = [f"<pad_vocab_{i}>" for i in range(pad_size)]
            tokenizer.add_tokens(pad_tokens)
        
        tokenizer.save_pretrained(config.tokenizer_path_padded)
        print(f"Saved padded tokenizer to {config.tokenizer_path_padded} with vocab size {len(tokenizer)}")
    else:
        print("'\\n' already in tokenizer.")
    
    # Reload to be safe
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path_padded)

    # 2. Process Datasets
    data_dir_base = "/mnt/project/kchat/edw/nmt_data/Data-Final-260414"
    
    dir_types = {
        "en-vi": ["<translate-en-vi>"],
        "vi-en": ["<translate-vi-en>"],
        "pair": ["<translate-en-vi>", "<translate-vi-en>"]
    }
    
    processed_datasets = []
    
    for dtype, directions in dir_types.items():
        sub_dir = os.path.join(data_dir_base, dtype)
        if not os.path.exists(sub_dir):
            continue
            
        hf_paths = find_hf_datasets(sub_dir)
        for hf_path in hf_paths:
            print(f"\\n--- Processing {hf_path} ---")
            ds = load_from_disk(hf_path)
            
            # Use train split if it exists, otherwise assume the dataset itself is the split
            if "train" in ds:
                ds = ds["train"]
                
            for direction in directions:
                print(f"Packing for direction: {direction}")
                
                remove_cols = ds.column_names
                map_func = partial(
                    map_and_pack_translation,
                    max_length=config.train_max_length_input,
                    tokenizer=tokenizer,
                    direction=direction
                )
                
                packed_ds = ds.map(
                    map_func, batched=True, batch_size=10000,
                    num_proc=16, remove_columns=remove_cols,
                    desc=f"Packing {direction}"
                )
                processed_datasets.append(packed_ds)
                
    if not processed_datasets:
        print("No datasets found to process!")
        return
        
    print(f"\\nConcatenating {len(processed_datasets)} datasets...")
    merged_dataset = concatenate_datasets(processed_datasets)
    
    # Shuffle the final train dataset
    merged_dataset = merged_dataset.shuffle(seed=config.seed)
    total_records = len(merged_dataset)
    print(f"\\n🚀 TOTAL PACKED TRAIN RECORDS: \\033[92m{total_records:,}\\033[0m")
    
    # 3. Save to TFDS
    print("Converting to TFDS format...")
    tfds = merged_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=True,
        batch_size=config.compress_batch,
        drop_remainder=True,
        num_workers=32,
    )
    tfds = tfds.map(cast_dtypes, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Save the TFDS
    save_path = f"{config.tfds_path.as_posix()}_train_packed"
    print(f"💾 Saving Packed TRAIN TFDS to {save_path}")
    tfds.save(save_path)
    print("✅ Training Data preparation complete!")

if __name__ == "__main__":
    main()
