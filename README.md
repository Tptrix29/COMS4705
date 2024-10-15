# COMS4705
Homework for Natural Language Processing course in Columbia University.

## Contents
- Homework 1: N-gram model
    - sentence generation
    - perplexity calculation: pp = 2^{-l}, l = \frac{1}{M} \sum_{i=1}^{N} s_i,  s_i = \frac{1}{n_i} \sum_{j=1}^{n_i} -\log_2 p(w_{i,j} | w_{i,j-1}, w_{i,j-2})
    - smoothing method: Laplace, Interpolation
- Homework 2: 
    
    **Metadata: **
    - conll_reader.py: data structures to represent a dependency tree, as well as functionality to read and write trees in the CoNLL-X format (explained below). 
    - get_vocab.py: extract a set of words and POS tags that appear in the training data. This is necessary to format the input to the neural net (the dimensionality of the input vectors depends on the number of words). 
    - extract_training_data.py: extracts two numpy matrices representing input output pairs (as described below). You will have to modify this file to change the input representation to the neural network.
    - train_model.py: specify and train the neural network model. This script writes a file containing the model architecture and trained weights. 
    - decoder.py: uses the trained model file to parse some input. For simplicity, the input is a CoNLL-X formatted file, but the dependency structure in the file is ignored. Prints the parser output for each sentence in CoNLL-X format. 
    - evaluate.py: this works like decoder.py, but instead of ignoring the input dependencies it uses them to compare the parser output. Prints evaluation results. 


## Usage
- Homework 1: 
```python
cd hw1
python -i trigram_model.py hw1_data/brown_train.txt hw1_data/brown_test.txt
```
- Homework 2: 
```python
cd hw2_torch
# generate vocab
python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
# extract training data
python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
# train model
python train_model.py data/input_train.npy data/target_train.npy data/model.pt
python train_model_gpu.py data/input_train.npy data/target_train.npy data/model.pt
# decode
python decoder.py data/model.pt data/dev.conll
# evaluate
python evaluate.py data/model.pt data/dev.conll
```