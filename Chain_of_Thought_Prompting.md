# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models 

Link to Original Paper \- [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

<img width="724" height="352" alt="image" src="https://github.com/user-attachments/assets/1157c05f-0300-4f7a-8ecc-2cfbd0dbb80d" />


# The Problem Definition: 

Large language models can perform many tasks via few-shot prompting, but they often struggle with complex reasoning tasks such as arithmetic, commonsense reasoning, and symbolic reasoning. Standard prompting methods directly map input to output without explicitly modeling intermediate reasoning steps.

# Issues with the Previous Methods:

Traditional few-shot prompting provides input-output examples but does not include intermediate reasoning steps. As a result:

1. Models tend to produce shallow or incorrect answers for multi-step problems.  
2. Performance degrades significantly on tasks requiring compositional or step-by-step reasoning.  
3. Smaller models especially fail to generalize reasoning beyond surface patterns.

These approaches do not explicitly encourage structured reasoning before generating the final answer.

# Key Innovation:

The paper introduces **Chain-of-Thought (CoT) prompting**, where examples in the prompt include intermediate reasoning steps along with the final answer.By providing step-by-step reasoning demonstrations, sufficiently large LLMs are able to generate their own reasoning chains, significantly improving performance on complex reasoning tasks.

# Core Explanation:

Chain-of-Thought prompting works by modifying few-shot prompts to include reasoning traces.

Instead of:

**Q → A**

The prompt becomes:

**Q → Step 1 → Step 2 → … → Final Answer**

When large models (e.g., 100B+ parameters) are given a few examples with reasoning chains, they learn to:

1. Generate intermediate reasoning steps.  
2. Decompose complex problems into smaller sub-problems.  
3. Arrive at more accurate final answers.

The improvement is especially strong on tasks such as \- Arithmetic reasoning, Commonsense reasoning, Symbolic reasoning. The paper also shows that **model scale matters**, smaller models do not benefit as much, while larger models show emergent reasoning abilities when prompted with CoT.

# Limitations:

1. **Dependence on Model Scale:**  
   Chain-of-Thought prompting works significantly better for large models. Smaller models show limited gains.  
2. **Prompt Sensitivity:**  
   Performance depends heavily on how the reasoning examples are written.  
3. **Longer Outputs & Higher Cost:**  
   Generating reasoning chains increases token usage and inference cost.  
4. **Not Guaranteed to Be Correct:**  
   Models may produce fluent but logically incorrect reasoning steps.

# Research Questions I would like to explore further: 

1. Can Chain-of-Thought reasoning be automatically optimized or learned instead of manually engineered in prompts?  
2. How can we verify or correct generated reasoning chains to prevent logically consistent but incorrect solutions?

