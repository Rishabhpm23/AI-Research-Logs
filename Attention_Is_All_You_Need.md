# Attention is All You Need 
Link to Original Paper - https://arxiv.org/abs/1706.03762

<img width="460" height="651" alt="image" src="https://github.com/user-attachments/assets/26f96376-5a9d-442d-b35c-a4c5bd7eb8a6" />


# The Problem Definition:

This paper introduced a neural network architecture for sequence transduction tasks that eliminates Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) entirely and relies solely on self-attention mechanisms to model global dependencies efficiently and enable parallelization in training.

# Issues with the Previous Methods:

RNNs, Long Short Term Memory (LSTMs) and gated RNNs, were the dominant approaches for sequence modeling and transduction problems. Many attempts to push boundaries of RNNs were made and significant improvements in computational efficiency, conditional computation and model performances were made. but , the inherent sequential nature made it impossible for parallelization within training examples, which was important at longer sequence lengths as memory constraints limit batching across examples.

# Key Innovation:

The “Transformers” allows more parallelization which reached SOTA in translation in those times. It allowed all tokens in a sequence to be processed parallely, unlike RNNs. This resulted in a significant decrease in the training time and helped capture long-range dependencies effectively. 

# Core Explanation 
“Attention” in “Attention is All You Need” refers to the model’s ability to dynamically emphasise different parts of the input text, and focus on the most relevant parts based on the task. The two main concepts introduced in the paper are Scaled Dot-Product Attention and Multi-Head Attention.

<img width="597" height="321" alt="image" src="https://github.com/user-attachments/assets/f7f2d0f9-37f9-4e3f-a2c2-9cfcc4bee15c" />


## Scaled Dot-Product Attention

This type of attention mechanism involves calculating the attention weights using scaled dot-product. The dot-product measures the similarity between the vectors, and scaling prevents the attention weights from becoming very large. 

This function takes three matrices as input: 

- The query matrix (Q):  Represents the current token asking, “what should i pay attention to ?”.  
- The key matrix (K): represents each tokens label, acting as “what information do I offer that might be relevant ?”.  
- The value matrix (V): Represents the actual content of the word that is weighted by the attention scores.

## Multi-Head Attention

This is a type of attention mechanism that uses multiple scaled dot-product attentions in parallel. Every dot-product attention head has its own set of Q, K, V matrices. This enables the model to attend to different parts of the input data in different ways. The output of each scaled dot-product is concatenated and that becomes the output of the multi-head attention function.


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

<img width="541" height="38" alt="image" src="https://github.com/user-attachments/assets/190a5aa0-9c2f-4ed8-938a-04fd9f72fd81" />


Learning rate increases linearly for the first 4000 steps (warmup phase). After warmup, it decays proportionally to the inverse square root of the step number. 

# Limitations

- **Large data requirements** (\~4.5M sentence pairs for English-German and \~36M for English-French)  
- **High computational cost** (Base model: 12 hours on 8 NVIDIA P100 GPUs , Big model: 3.5 days on 8 P100 GPUs)  
- **Fixed context window** (The model processes sequences of fixed maximum length and cannot easily handle documents longer than the maximum context window without truncation)

# Research Questions I would like to explore further 

- What architectural modifications are possible that could reduce the training time without compromising performance ?  
- How effective are data augmentation techniques for Transformers in low-resource settings?
