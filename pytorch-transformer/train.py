import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from dataset import BilingualDataset
from model import TransformerModel, build_transformer_model


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer'] = '../tokenizer/tokenizer_{0}.json'
    tokenizer_path = Path(config["tokenizer_dir"].format(lang))
    if not Path.exists(tokenizer_path):
        # print("Tokenizer not found, building tokenizer")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            # vocab_size=config["vocab_size"],
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        # print("Tokenizer found, loading tokenizer")
        tokenizer = Tokenizer.from_file(tokenizer_path)


def get_ds(config):
    ds_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # keep 90% of the dataset for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length source: {max_len_src}, max length target: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer_model(
        vocab_src_len,
        vocab_tgt_len,
        config["d_model"],
        config["d_ff"],
        config["num_heads"],
        config["num_layers"],
        config["dropout"],
        config["max_seq_len"],
    )
    return model 


