import torch
import torch.nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer'] = '../tokenizer/tokenizer_{0}.json'
    tokenizer_path = Path(config["tokenizer_dir"].format(lang))
    if not Path.exists(tokenizer_path):
        print("Tokenizer not found, building tokenizer")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            # vocab_size=config["vocab_size"],
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        )
        tokenizer.train_from_iterator(ds["train"]["text"], trainer=trainer)
        tokenizer.save(tokenizer_path)
