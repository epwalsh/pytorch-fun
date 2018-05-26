"""
Defines a names classification models.
"""

import torch.nn as nn
from torch.nn import LSTM


class CharLSTM(nn.Module):
    """
    Character-level LSTM classifier.

    Parameters
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    hidden_size : int
        The number of features in the hidden state of the LSTM cells.

    bidirectional : bool
        If true, becomes a bidirectional LSTM.

    lstm_layers : int
        The number of cell layers in the LSTM.

    Attributes
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    hidden_size : int
        The number of features in the hidden state of the LSTM cells.

    bidirectional : bool
        If true, becomes a bidirectional LSTM.

    lstm_layers : int
        The number of cell layers in the LSTM.

    rnn : :obj:`torch.nn.Module`
        The LSTM layer.

    linear : :obj:`torch.nn.Module`
        The linear output layer.

    softmax : :obj:`torch.nn.Module`
        The softmax transformation layer.

    """
    def __init__(self, n_chars, hidden_size, output_size,
                 bidirectional=True, lstm_layers=1):
        super(CharLSTM, self).__init__()
        self.n_chars = n_chars
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers

        # LSTM layer.
        self.rnn = LSTM(self.n_chars, self.hidden_size, self.lstm_layers,
                        batch_first=True, bidirectional=self.bidirectional)

        # Linear output layer.
        directions = 2 if self.bidirectional else 1
        self.linear = nn.Linear(
            self.hidden_size * directions * self.lstm_layers, self.output_size)

        # Softmax output transformation layer.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word):
        """
        Make a forward pass through the network.

        Parameters
        ----------
        word : :obj:`torch.Tensor`
            Tensors of shape `[word_length x n_chars]`.

        Returns
        -------
        :obj:`torch.Tensor`
            The last hidden state:
            `[1 x (layers x directions x hidden_size)]`

        """
        # First run through the LSTM layer.
        _, state = self.rnn(word.unsqueeze(0))

        # `[(layers x directions) x 1 x hidden_size]`
        hidden = state[0]

        # Get rid of batch_size dimension.
        # `[(layers x directions) x hidden_size]`
        hidden = hidden.squeeze()

        # Concatenate forward/backward hidden states.
        # Changes to `[1 x (layers x directions x hidden_size)]`.
        hidden = hidden.view(-1).unsqueeze(0)

        # Run through linear layer:
        # `[1 x (output)]`.
        output = self.linear(hidden)

        # Apply softmax.
        output = self.softmax(output)

        return output
