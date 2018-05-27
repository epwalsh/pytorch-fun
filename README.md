# pytorch-fun

Small, independent PyTorch examples that can be trained within a few minutes on CPU.
Datasets included, batteries sold separately.

To train an example, just run one of the available `train_*.py` files within the corresponding subdirectory.
The only requirement is [PyTorch](https://pytorch.org/).

## List of examples by task type

| Task | Example | Description |
|------|--------------|-------------|
| `multi-class classification` | `./name_classification/` | Classification of surnames by their language of origin using a simple Bi-LSTM to linear architecture. The dataset comes from the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html). |
| `binary classification` | `./loan_default/` | Predicting whether or not someone defaults on their loan using logistic regression. The dataset is the simulated dataset `Default` from the book [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/index.html). |
