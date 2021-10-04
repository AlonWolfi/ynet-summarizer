import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 32
        self.embedding_dim = 32
        self.num_layers = 1

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, randomize=False):
        embed = self.embedding(x)
        batch_size = x.shape[0]
        if randomize:
            prev_state = self.random_state(batch_size)
        else:
            prev_state = self.init_state(batch_size)
        output, state = self.lstm(embed, prev_state)

        output = output[:, [-1], :]
        logits = self.fc(output)
        logits = logits.transpose(1, 2)[:, :, 0]
        return logits, (prev_state, state)

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.lstm_size),
                torch.zeros(self.num_layers, batch_size, self.lstm_size))

    def random_state(self, batch_size):
        return (torch.randn(self.num_layers, batch_size, self.lstm_size),
                torch.randn(self.num_layers, batch_size, self.lstm_size))
