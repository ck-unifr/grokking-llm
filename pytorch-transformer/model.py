import torch
import torch.nn as nn


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int, max_len: int = 512):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # super(InputEmbedding, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size, d_model)
        # self.position_embedding = nn.Embedding(max_len, d_model)
        # self.dropout = nn.Dropout(0.1)
