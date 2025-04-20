import torch
import torch.nn as nn
import math 

"""
https://www.youtube.com/watch?v=ISNdQcPhsts&t=5s
"""


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        # super(InputEmbedding, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size, d_model)
        # self.position_embedding = nn.Embedding(max_len, d_model)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
        # x = self.word_embedding(x) + self.position_embedding(torch.arange(x.size(1)).unsqueeze(0).expand_as(x))
        # x = self.dropout(x)
        # return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.seq_len = seq_len

        # Create a matrix of size (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # Apply the sine to the even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cosine to the odd indices in the array 
        pe[:, 1::2] = torch.cos(position * div_term) 

        # Add a batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe) # Register the buffer so that it is not a parameter
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the positional encoding to the input tensor
        x = x + (self.pe[:, :x.shape(1), :]).requires_grad_(False)
        # Apply dropout
        return self.dropout(x)

    def _slow_forward(self, *input, **kwargs):
        return super()._slow_forward(*input, **kwargs)