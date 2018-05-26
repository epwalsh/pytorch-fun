#!/usr/bin/env python
"""
Trains the LSTM character-level classifier to the names dataset.
"""

import random

import torch

from models import CharLSTM


N_LANGS = 18
HIDDEN_SIZE = 50


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
        tgt = torch.zeros(N_LANGS)
        tgt[lang] = 1.
        data.append((inp, tgt))

    random.shuffle(data)

    return data, characters


def main():
    """Main func to run."""
    print("Gathering data...")
    train, char_dict = gather_data()

    print("Initializing model...")
    model = CharLSTM(len(char_dict), HIDDEN_SIZE, N_LANGS)

    print(model(train[0][0]))


if __name__ == "__main__":
    main()
