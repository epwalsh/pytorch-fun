#!/usr/bin/env python
# encoding: utf-8

"""
Trains the LSTM character-level classifier to the names dataset.
"""

import random

import torch

from models import CharLSTM


HIDDEN_SIZE = 50
EPOCHS = 2
PLOT_EVERY = 1000
LANGS = {
    0: "Arabic",
    1: "Chinese",
    2: "Czech",
    3: "Dutch",
    4: "English",
    5: "French",
    6: "German",
    7: "Greek",
    8: "Irish",
    9: "Italian",
    10: "Japanese",
    11: "Korean",
    12: "Polish",
    13: "Portuguese",
    14: "Russian",
    15: "Scottish",
    16: "Spanish",
    17: "Vietnamese",
}
N_LANGS = len(LANGS)


def gather_data():
    """Gather raw data and build character dictionary."""
    characters = {}
    raw_data = []
    with open("./data.tsv", "r") as datafile:
        for line in datafile.readlines():
            name, lang = line.split('\t')
            raw_data.append((name, int(lang)))
            for char in name:
                characters.setdefault(char, len(characters))

    n_chars = len(characters)
    data = []
    for name, lang in raw_data:
        inp = torch.zeros(len(name), n_chars)
        for i, char in enumerate(name):
            inp[i][characters[char]] = 1.
        tgt = torch.tensor([lang], dtype=torch.long)
        data.append((name, inp, tgt))

    random.shuffle(data)

    return data, characters


def main():
    """Main func to run."""
    print("Gathering data...")
    train, char_dict = gather_data()

    print("Initializing model...")
    model = CharLSTM(len(char_dict), HIDDEN_SIZE, N_LANGS)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    print("Training model...")
    for iteration in range(EPOCHS):
        print("Epoch {:d}".format(iteration))
        total_loss = 0
        for i, example in enumerate(train):
            raw_name, name, lang = example
            model.zero_grad()
            output = model(name)
            loss = criterion(output, lang)
            loss.backward()
            optimizer.step()
            total_loss += loss
            if i > 0 and i % PLOT_EVERY == 0:
                _, top_lang_i = output.topk(1)
                top_lang_i = top_lang_i[0].item()
                guess = LANGS[top_lang_i]
                actual = LANGS[lang[0].item()]
                correct = "✓" if guess == actual else "✗ ({:s})".format(actual)
                print("{:d}: {:s} / {:s} {:s}"
                      .format(i, raw_name, guess, correct))
        print("Epoch loss: {:f}".format(total_loss))


if __name__ == "__main__":
    main()
