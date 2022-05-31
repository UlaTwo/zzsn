from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

from enum import Enum


@dataclass
class ModelConfig:
    epochs: int
    batch_size: int
    embedding_size: int
    num_hidden: int
    num_layers: int
    seq_len: int
    dropout: float
    learning_rate: float
    clip: float

    def get_dict(self):
        config = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'embedding_size': self.embedding_size,
            'num_hidden': self.num_hidden,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'seq_len': self.seq_len,
            'learning_rate': self.learning_rate,
            'clip': self.clip
        }

        return config

class ModelType(Enum):
    GRU = 1
    LSTM = 2
    R_TRANSFORMER = 3


class RNNModel(nn.Module):
    def __init__(self, model_type, vocab_size, embedding_size, hidden_units, num_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.model_type = model_type

        if self.model_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_units,
                            num_layers, dropout=dropout)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_units,
                            num_layers, dropout=dropout)
        else:
            raise NotImplementedError()
        self.linear = nn.Linear(hidden_units, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.vocab_size = vocab_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers

    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        decoded = self.linear(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.model_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_units),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_units))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_units)