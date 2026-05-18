"""Train a 20k BPE tokenizer on PhoMT (en+vi) and save at
config.tokenizer_path_padded, with project special tokens and vocab padded
to a multiple of 8.
"""
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from config import config

VOCAB_SIZE = 20000
SPECIAL = ["<pad>", "<unk>", "<s>", "</s>",
           "<translate-en-vi>", "<translate-vi-en>",
           "<tone-teen>", "<json>", "<cls>", "<info_ext>", "\n"]


def text_iterator(ds):
    for ex in ds:
        t = ex.get("translation") or ex
        if isinstance(t, dict):
            en, vi = t.get("en"), t.get("vi")
            if en: yield str(en)
            if vi: yield str(vi)


def main():
    out_dir = config.tokenizer_path_padded
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[ok] tokenizer already exists at {out_dir}")
        return

    print(f"[bootstrap] loading {config.dataset_name} (train split)")
    ds = load_dataset(config.dataset_name, split="train")
    print(f"[bootstrap] {len(ds):,} pairs -> {2 * len(ds):,} lines")

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"[bootstrap] training BPE vocab_size={VOCAB_SIZE}")
    tok.train_from_iterator(text_iterator(ds), trainer=trainer, length=2 * len(ds))

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>", unk_token="<unk>",
        bos_token="<s>", eos_token="</s>",
    )

    vocab_size = len(wrapped)
    if vocab_size % 8 != 0:
        pad_n = 8 - (vocab_size % 8)
        wrapped.add_tokens([f"<pad_vocab_{i}>" for i in range(pad_n)])
        print(f"[bootstrap] padded vocab by {pad_n} -> {len(wrapped)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    wrapped.save_pretrained(out_dir)
    print(f"[bootstrap] saved tokenizer ({len(wrapped)} tokens) to {out_dir}")


if __name__ == "__main__":
    main()
