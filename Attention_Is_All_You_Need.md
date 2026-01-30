# Attention is All You Need 


# The Problem Definition:

This paper introduced a neural network architecture for sequence transduction tasks that eliminates Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) entirely and relies solely on self-attention mechanisms to model global dependencies efficiently and enable parallelization in training.

# Issues with the Previous Methods:

RNNs, Long Short Term Memory (LSTMs) and gated RNNs, were the dominant approaches for sequence modeling and transduction problems. Many attempts to push boundaries of RNNs were made and significant improvements in computational efficiency, conditional computation and model performances were made. but , the inherent sequential nature made it impossible for parallelization within training examples, which was important at longer sequence lengths as memory constraints limit batching across examples.

# Key Innovation:

The “Transformers” allows more parallelization which reached SOTA in translation in those times. It allowed all tokens in a sequence to be processed parallely, unlike RNNs. This resulted in a significant decrease in the training time and helped capture long-range dependencies effectively. 

# Core Explanation 

# Experimental Setup 

## Datasets 

Transformers was trained on standard machine translation benchmark datasets:

### WMT 2014 English–German 

This dataset contains \~4.5 million sentence pairs. Tokenization on this was performed using Byte-Pair Encoding (BPE). The shared vocab size was 37000 tokens.

### WMT 2014 English–French 

This dataset contains \~36 million sentence pairs. Tokenization on this was performed using Byte-Pair Encoding (BPE). The shared vocab size was 32000 tokens.

## Hardware

Trained on one machine with 8 NVIDIA P100 GPUs

## Optimizer

Adam Optimizer (β₁ \= 0.9,  β₂ \= 0.98,  ε \= 10⁻⁹ )

## Learning Rate

The Transformers has a custom learning rate which is defined as:   
![Learning Rate Schedule](https://latex.codecogs.com/png.latex?\Large%20lrate%20=%20d_{model}^{-0.5}\cdot\min(step^{-0.5},\ step\cdot warmup\_steps^{-1.5}))
Learning rate increases linearly for the first 4000 steps (warmup phase). After warmup, it decays proportionally to the inverse square root of the step number. 

# Limitations

# Research Questions I would like to explore further 
