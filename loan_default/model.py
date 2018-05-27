"""
Defines a logistic regression model for binary classification.
"""

import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Logistic Regression model for binary classification.

    Parameters
    ----------
    n_feats : int
        The number of input features.

    Attributes
    ----------
    n_feats : int
        The number of input features.

    """

    def __init__(self, n_feats):
        super(LogisticRegression, self).__init__()
        self.n_feats = n_feats
        self.linear = nn.Linear(self.n_feats, 1)

    def forward(self, inputs):
        """
        Compute the logits of the input.

        Parameters
        ----------
        inputs : :obj:`torch.Tensor`
            The input of observations: `[batch_size x n_feats]`

        Returns
        -------
        :obj:`torch.Tensor`
            The logits: `[batch_size x 1]`.
        """
        # Compute the logits:
        # `[batch_size x 1]`
        output = self.linear(inputs)

        return output
