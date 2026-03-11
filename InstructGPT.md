# Paper Summary: Training Language Models to Follow Instructions with Human Feedback

**Authors:** Ouyang et al. (OpenAI)  
**Published:** 2022 | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)  
**Tags:** `RLHF` `Instruction Following` `Alignment` `InstructGPT` `PPO`

---

<img width="1094" height="613" alt="image" src="https://github.com/user-attachments/assets/c54d4df8-cddc-4e59-ae0a-aacc94612038" />


## 1. Problem Definition

Making language models bigger does **not** make them better at following user intent. Scaling just makes models better at predicting the next token — which is not the same thing as doing what the user actually wants. Basically, these models are not aligned with their users.

So the question this paper asks is: how do you take a giant pretrained model and actually make it follow instructions the way a real person would want?

---

## 2. Why Prior Methods Fail

Raw GPT-3 (175B parameters) is trained on internet text. It's extremely capable, but its objective — predict the next token — doesn't care about being helpful, honest, or safe. It'll complete prompts in ways that are plausible given the training data, not in ways that are actually useful to the person asking.

There was no mechanism to tell the model "this is the kind of response a user actually wants." You could prompt-engineer your way around it, but that only goes so far. The model has no concept of user intent baked into its weights.

---

## 3. Key Innovation

> Fine-tune GPT-3 with **human feedback** in a 3-step pipeline — supervised fine-tuning on demonstrations, reward model training on human rankings, and PPO-based RL against that reward model. The resulting models are called **InstructGPT**.

The punchline: 1–3B parameter InstructGPT outputs are **preferred over 175B GPT-3** outputs, despite having 100x fewer parameters. Size isn't everything — alignment is.

InstructGPT also shows clear improvements in **truthfulness** and a meaningful **reduction in toxic output generations** compared to the base model.

---

## 4. Core Method Explanation

The pipeline has 3 steps:

### Step 1 — Collect Demonstration Data & Train a Supervised Policy

```
[A prompt is sampled from the prompt dataset]
                    |
                    v
[A labeler demonstrates the desired output behaviour]
                    |
                    v
[This data is used to fine-tune GPT-3 with supervised learning]
```

- Start with a set of **labeler-written prompts** plus prompts submitted through the OpenAI API.
- Human labelers write out exactly what a good response looks like for each prompt.
- Use this data to fine-tune GPT-3 with **supervised learning** — straightforward imitation of the demonstrated behavior.
- This gives you a solid starting model, but it's limited by how much demonstration data you can collect.

### Step 2 — Collect Comparison Data & Train a Reward Model

```
[A prompt + several model outputs are sampled]
                    |
                    v
[A labeler ranks the outputs from best to worst]
                    |
                    v
[This data is used to train the reward model]
```

- For each prompt, sample several outputs from the supervised model.
- Labelers **rank** those outputs from best to worst — ranking is cheaper and more reliable than writing demonstrations from scratch.
- Train a **reward model (RM)** on these rankings — the RM learns to predict which outputs humans prefer.

### Step 3 — Optimize the Policy Against the Reward Model Using Reinforcement Learning

```
[A new prompt is sampled from the dataset]
                    |
                    v
            [The policy generates an output] <─────────────┐
                    |                                      |
                    v                                      |
    [The reward model calculates a reward for the output]  |
                    |                                      |
                    v                                      |
    [The reward is used to update the policy using PPO] ───┘
```

- Sample a new prompt from the dataset.
- The current policy generates an output.
- The reward model scores that output.
- The score is used to update the policy via **PPO (Proximal Policy Optimization)**.
- This loop runs continuously — the policy keeps improving against the reward signal.

The reward model is the key bridge here: it distills thousands of human preference judgments into a single differentiable signal that RL can optimize against.

---

## 5. Experimental Setup

- **Base model:** GPT-3 (with variants at 1.3B, 6B, and 175B parameters)
- **Data:** Labeler-written prompts + real OpenAI API prompts from users; human rankings collected for reward model training
- **Labelers:** Contractors trained to evaluate outputs for helpfulness, truthfulness, and harmlessness
- **RL algorithm:** PPO with a KL penalty term to prevent the policy from drifting too far from the original supervised model
- **Evaluation:** Human preference ratings (labelers and held-out users compare outputs head-to-head); also TruthfulQA, RealToxicityPrompts, and other benchmarks
- **Key result:** 1.3B InstructGPT is preferred over 175B GPT-3 by labelers. InstructGPT also hallucinates less and produces fewer toxic completions — without a significant drop on standard NLP benchmarks.

---

## 6. Limitations

- **Labeler bias is baked in.** The reward model learns what _these specific labelers_ prefer, which may not reflect what all users want. Different labeler pools could produce very different models.
- **"Alignment tax" on some benchmarks.** InstructGPT performs slightly worse on some academic NLP benchmarks compared to GPT-3. Optimizing for human preference can hurt narrow benchmark performance.
- **The reward model can be gamed.** PPO will find ways to produce outputs that score well on the RM without actually being better — reward hacking is a real risk, especially with long training runs.
- **Expensive to scale.** Every step requires human labor — writing demonstrations, doing rankings. This doesn't get easier as you scale the model up.
- **No formal definition of "helpful."** Labelers use their judgment, but "helpful" is subjective and context-dependent. The paper doesn't resolve this — it just assumes the labelers have reasonable taste.

---

_Summary written as part of the 45-Day Hard research plan._
