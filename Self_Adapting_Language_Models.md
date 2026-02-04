# Self-Adapting Language Models

Link to the Original Paper \- [https://arxiv.org/abs/2506.10943](https://arxiv.org/abs/2506.10943)

<img width="830" height="337" alt="image" src="https://github.com/user-attachments/assets/f0f10e52-a580-4a2b-bac9-2301ad42f6dc" />

# The Problem Definition: 

Large language models (LLMs) lack a mechanism to adapt their weights in response to new tasks, knowledge or examples. Self-Adapting LLMs (SEAL) enables LLMs to self-adapt by generating their own finetuning data & update directives.

# Issues with the Previous Methods:

If given a new task, LLMs consume and learn from task data as it is via finetuning or in-context learning. But that data might not be an optimal format for learning and current approaches do not allow models to develop strategies which will best transform & learn from the training data.

# Key Innovation:

SEAL allows LLMs to generate their own training data & finetuning directives in response to new inputs. This paper introduces a reinforcement learning algorithm that trains LLMs to generate “self-edits”. Hence known as Self-Adapting LLM (SEAL).

# Core Explanation:
<img width="350" height="237" alt="image" src="https://github.com/user-attachments/assets/69b81d3c-75e4-4516-aa58-144a5939949d" />

The SEAL framework works in two main steps:

1. **Inner Step (Learning):**  
    The Model takes a task, like a text passage and creates a self-edit. It then uses that self edit to update its knowledge through supervised finetuning (SFT), which adjusts the model's settings efficiently using Low-Rank Adaptation (LoRA).

2. **Outer Step (Improving):**  
    The model then tests how well it did on the task after updating. If it performs well, it rewards the self-edit & learns to make a similar one. This is done using ReSTEM (rejection sampling \+ SFT). EM can be viewed as an expectation-maximization procedure.

The overall approach is like teaching the model to teach itself. SEAL uses the LLM’s own ability to write text & guide its learning.

# Limitations:

1. **Forgetting old knowledge:**  
   When SEAL learns new things, it may forget older information.

2. **High Computation Needs:**  
   Training SEAL with RL takes a lot of compute power.

3. **Scaling SEAL:**  
   The researchers tested using smaller models, & its not clear how it would work with very big LLMs.

# Research Questions I would like to explore further: 

1. How can SEAL mitigate catastrophic forgetting while continuously generating self-edits?  
2. How does SEAL scale to large foundation models under realistic compute constraints?
