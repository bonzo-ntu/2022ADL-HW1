from typing import Dict

import torch
from torch.nn import Embedding
from torch import nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        batch_size: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        # initializer
        self._num_layers, self._hidden_size, self._dropout, self._bidirectional, self._num_class = (
            num_layers,
            hidden_size,
            dropout,
            bidirectional,
            num_class,
        )
        self._batch_size = batch_size

        # neural net
        self.rnn = nn.GRU(
            300,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
            batch_first=True,
            bidirectional=self._bidirectional,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=self._batch_size)
        self.dropout = nn.Dropout(self._dropout)
        self.fc1 = nn.Linear(self._hidden_size * 2, self._hidden_size // 2)
        self.fc2 = nn.Linear(self._hidden_size // 2, self._num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    printyet = False

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        out = self.embed(batch)
        out, _ = self.rnn(out)
        out = self.batch_norm(out)
        out = (nn.LeakyReLU(0.1))(out)
        out = self.dropout(out[:, -1, :].view(out.size(0), -1))
        out = self.fc1(out)
        out = (nn.LeakyReLU(0.1))(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class SeqTagger(SeqClassifier):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        batch_size: int,
    ) -> None:
        super().__init__(embeddings, hidden_size, num_layers, dropout, bidirectional, num_class, batch_size)
        # TODO: model architecture

        self.fc1 = nn.Linear(self._hidden_size * 2, self._hidden_size)
        self.fc2 = nn.Linear(self._hidden_size, self._hidden_size // 2)
        self.fc3 = nn.Linear(self._hidden_size // 2, self._num_class)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        out = self.embed(batch)
        out, _ = self.rnn(out)
        out = self.batch_norm(out)
        out = (nn.LeakyReLU(0.1))(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = (nn.LeakyReLU(0.1))(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = (nn.LeakyReLU(0.1))(out)
        out = self.dropout(out)
        out = self.fc3(out)

        return out
