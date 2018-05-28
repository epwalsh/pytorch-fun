#!/usr/bin/env python
# encoding: UTF-8
# pylint: disable=not-callable

"""Train the logistic regression model on the 'Default' dataset."""

import random

import torch
import torch.nn as nn

from model import LogisticRegression


N_FEATS = 3
LOG_EVERY = 500
EPOCHS = 5


def gather_data():
    """
    Read data into tensors.
    """
    out = []
    with open("./data.csv", "r") as datafile:
        # Skip header.
        _ = next(datafile)
        for line in datafile:
            line = line.rstrip().split(",")
            tgt = torch.tensor([[1. if line[0] == "Yes" else 0.]])
            inp = torch.zeros(1, N_FEATS)
            inp[0][0] = 1. if line[1] == "Yes" else 0.
            inp[0][1] = float(line[2])
            inp[0][2] = float(line[3])
            out.append((inp, tgt))
    return out


def main():
    """
    Train the model and print progress.
    """
    data = gather_data()
    model = LogisticRegression(N_FEATS)
    optimizer = torch.optim.Adam(model.parameters())

    for iteration in range(EPOCHS):
        print("Epoch {:d}".format(iteration + 1))

        # Shuffle data.
        random.shuffle(data)

        total_loss = 0
        running_loss = 0
        n_examples = len(data)

        # Loop through examples in data.
        for i, example in enumerate(data):
            inp, tgt = example

            # Zero out the gradient.
            model.zero_grad()

            # Make a forward pass, i.e. compute the logits.
            logits = model(inp)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, tgt)

            # Compute gradient and take step.
            loss.backward()
            optimizer.step()

            total_loss += loss
            running_loss += loss

            # Print progress.
            if (i + 1) % LOG_EVERY == 0:
                guess = logits[0].item() > 0
                actual = tgt[0].item() == 1
                correct = "✓" if guess == actual else "✗ ({:})".format(actual)
                print("({:d} / {:d}) Loss: {:.5f}"
                      .format(i + 1, n_examples, running_loss))
                print(" => {:} {:}"
                      .format(guess, correct))
                running_loss = 0

        print("Epoch loss: {:f}\n".format(total_loss))


if __name__ == "__main__":
    main()
