import sys
import os
from functools import partial

# Prevent Rust Tokenizer deadlocks in multiprocessing
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.extend(['/mnt/data/edw_2'])
import tensorflow as tf

# Disable GPU to avoid consuming memory JAX will need later
tf.config.set_visible_devices([], 'GPU')

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from config import config


def preprocess_function_translation(examples, max_length_input, max_length_labels, tokenizer,
                                    direction="<translate-en-vi>"):
    direction_map = {
        "<translate-en-vi>": ("en", "vi"),
        "<translate-vi-en>": ("vi", "en"),
    }
    source_lang, target_lang = direction_map[direction]

    if 'translation' in examples:
        inputs = [f"{direction} " + str(ex[source_lang]) for ex in examples['translation']]
        targets = [str(ex[target_lang]) for ex in examples['translation']]
    else:
        inputs = [f"{direction} " + str(ex) for ex in examples[source_lang]]
        targets = [str(ex) for ex in examples[target_lang]]

    model_inputs = tokenizer(
        inputs, max_length=max_length_input, truncation=True,
        padding="max_length", return_tensors="np"
    )
    labels = tokenizer(
        targets, max_length=max_length_labels, truncation=True,
        padding="max_length", return_tensors="np"
    ).input_ids

    model_inputs["labels"] = labels
    return model_inputs


def main():
    print(f'Python {sys.version} on {sys.platform}')

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path_padded)
    directions = {"<translate-en-vi>": ("en", "vi"), "<translate-vi-en>": ("vi", "en")}

    print(f"Loading Benchmark Dataset: {config.dataset_name}...")
    dataset = load_dataset(config.dataset_name)

    splits = {'train': 'train', 'validation': 'validation', 'test': 'test'}
    if 'validation' not in dataset and 'val' in dataset:
        splits['validation'] = 'val'

    def is_valid_row(example):
        if 'translation' in example and isinstance(example['translation'], dict):
            en_text = example['translation'].get('en')
            vi_text = example['translation'].get('vi')
        else:
            en_text = example.get('en')
            vi_text = example.get('vi')

        return bool(en_text) and bool(vi_text) and str(en_text).strip() != "" and str(vi_text).strip() != ""

    def cast_dtypes(features):
        features['input_ids'] = tf.cast(features['input_ids'], tf.int32)
        features['attention_mask'] = tf.cast(features['attention_mask'], tf.int32)
        features['labels'] = tf.cast(features['labels'], tf.int32)
        return features

    for split_name, split_key in splits.items():
        if split_key not in dataset:
            continue

        print(f"\n{'-' * 50}\n--- Processing Benchmark Split: {split_name.upper()} ---\n{'-' * 50}")
        ds_split = dataset[split_key]

        original_len = len(ds_split)
        ds_split = ds_split.filter(is_valid_row, num_proc=16, desc=f"Filtering bad rows from {split_name}")
        dropped = original_len - len(ds_split)
        if dropped > 0:
            print(f"[\033[93mWarning\033[0m] Dropped {dropped} corrupt/empty rows from {split_name}.")

        remove_cols = ds_split.column_names

        combined = DatasetDict()
        for direction in directions.keys():
            map_func = partial(
                preprocess_function_translation,
                max_length_input=config.train_max_length_input,
                max_length_labels=config.train_max_length_output,
                tokenizer=tokenizer,
                direction=direction
            )

            tokenized_dataset = ds_split.map(
                map_func, batched=True, num_proc=16, remove_columns=remove_cols,
                desc=f"Preprocessing {split_name.upper()} ({direction})"
            )
            combined[direction] = tokenized_dataset

        # 🌟 UPGRADE: Now we split BOTH validation and test sets purely by direction
        if split_name in ['test', 'validation']:
            for direction_tag, single_direction_ds in combined.items():
                suffix = "en_vi" if "en-vi" in direction_tag else "vi_en"
                print(f"\n✅ TOTAL {split_name.upper()} RECORDS ({suffix.upper()}): \033[92m{len(single_direction_ds):,}\033[0m")

                tfds = single_direction_ds.to_tf_dataset(
                    columns=["input_ids", "attention_mask", "labels"],
                    shuffle=False,
                    batch_size=config.eval_batch_size,
                    drop_remainder=False,
                    num_workers=32,
                )
                tfds = tfds.map(cast_dtypes, num_parallel_calls=tf.data.AUTOTUNE)

                save_path = f"{config.tfds_path.as_posix()}_{split_name}_{suffix}"
                print(f"💾 Saving {split_name.upper()} ({suffix.upper()}) TFDS to {save_path}")
                tfds.save(save_path)

        else:
            # Training set remains perfectly mixed and shuffled
            merged_dataset = concatenate_datasets(list(combined.values()))
            total_records = len(merged_dataset)
            print(f"\n🚀 TOTAL {split_name.upper()} RECORDS (Both Directions Mixed): \033[92m{total_records:,}\033[0m")

            tfds = merged_dataset.to_tf_dataset(
                columns=["input_ids", "attention_mask", "labels"],
                shuffle=True,
                batch_size=config.compress_batch,
                drop_remainder=True,
                num_workers=32,
            )
            tfds = tfds.map(cast_dtypes, num_parallel_calls=tf.data.AUTOTUNE)

            save_path = f"{config.tfds_path.as_posix()}_{split_name}"
            print(f"💾 Saving {split_name.upper()} TFDS to {save_path}")
            tfds.save(save_path)

    print("\n✅ Benchmark Data preprocessing complete!")


if __name__ == "__main__":
    main()