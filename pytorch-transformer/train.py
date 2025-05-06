import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset, random_split

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

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


def train_model(config):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    ).to(device)
    # tensorboard
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        init_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dir"])
        global_step = state["global_step"]

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1,
    ).to(device)

    for epoch in range(init_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, seq_len)
            
            # run the tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)    
            decoder_output = model.decode(decoder_input, encoder_output, decoder_mask) # (batch_size, seq_len, d_model)
            
