# Constitutional AI — Harmlessness from AI Feedback

**Authors:** Yuntao Bai et al. (Anthropic)  
**Published:** December 2022 | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)  
**Tags:** `AI Alignment` `RLHF` `RLAIF` `Harmlessness` `Self-Supervision`

---

<img width="1426" height="765" alt="image" src="https://github.com/user-attachments/assets/a4fec8e7-b15e-4b13-87a4-c6e2ddc28f7c" />


## 1. Problem Definition

We want AI assistants that are helpful, honest, **and** harmless — even as they get more capable. The challenge is: how do you train a model to be harmless _without_ needing thousands of humans to label every single harmful output?

That's the core question this paper answers.

---

## 2. Why Prior Methods Fail

Before this paper, the standard approach was **RLHF** (Reinforcement Learning from Human Feedback) — the same pipeline used to build InstructGPT. Here's how it works at a high level (from my reading notes):

> **Step 1 — SFT:** A prompt is sampled → a human labeler demonstrates the desired output → this data fine-tunes GPT-3 with supervised learning.  
> **Step 2 — Reward Model:** A prompt + several model outputs are sampled → a labeler ranks them best to worst → this trains a reward model.  
> **Step 3 — RL:** A new prompt is sampled → the policy generates an output → the reward model scores it → the score updates the policy via PPO. Loop.

The problem with standard RLHF for harmlessness:

- **It's expensive.** You need tens of thousands of human labels _just for harmlessness_.
- **It makes models evasive.** When harmlessness is rewarded by crowd workers, models learn that "I can't answer that" is a safe bet — which tanks helpfulness.
- **The objectives are opaque.** No one can read 50,000 preference labels and understand what the model actually learned.

There's also a real tension: training harder for harmlessness tends to make the model _less_ helpful, and vice versa.

---

## 3. Key Innovation

> Train a model to be harmless using **only a short list of written principles** — no human feedback labels for harmlessness at all. Let the AI critique and revise its own outputs (supervised stage), then use AI-generated preference labels to do RL (RLAIF stage).

That's it. The model does the heavy lifting. Humans just write the rules.

---

## 4. Core Method Explanation

The process has two stages:

### Stage 1 — Supervised Learning (SL-CAI): Critique → Revise → Fine-tune

Start with a helpful-only RLHF model. Feed it a harmful prompt. It gives a bad answer. Then:

1. Ask the model to **critique** its own response using a principle from the "constitution" (e.g., _"Identify ways your response is harmful, toxic, or illegal"_).
2. Ask the model to **revise** the response based on the critique.
3. Repeat this critique-revision loop several times with different principles.
4. Fine-tune a pretrained LM on the final revised responses.

This gets the model "on-distribution" before RL starts — it reduces the need for random exploration.

### Stage 2 — RL from AI Feedback (RLAIF): AI Labels → Preference Model → PPO

1. Take the SL-CAI model and generate **pairs** of responses to harmful prompts.
2. Ask a separate feedback model to pick the _less harmful_ response using a constitutional principle — formatted as a multiple choice question.
3. Use these AI-generated labels to train a **preference model (PM)**.
4. Fine-tune the SL-CAI model using PPO against this PM.

One clever trick: they use **Chain-of-Thought prompting** on the feedback model. Asking it to _reason step-by-step_ before choosing significantly improves label quality. They also clamp CoT probabilities to the 40–60% range to prevent overconfident labels from destabilizing training.

---

## 5. Experimental Setup

- **Models:** 52B parameter LMs (and ablations across sizes from ~1B to 52B)
- **Data:**
  - 42,496 human-written + 140,335 model-generated red team prompts for harmlessness
  - 135,296 human helpfulness prompts (human labels kept for helpfulness)
  - 16 different constitutional principles, randomly sampled at each revision step
- **Evaluation:** Elo scores computed from crowdworker comparison tests (A/B testing model responses)
- **Baselines:** Helpful-only RLHF, HH (helpful + harmless) RLHF, SL-CAI, RL-CAI, RL-CAI with CoT
- **Key result:** RL-CAI with CoT achieves a **Pareto improvement** over standard RLHF — it's both more helpful _and_ more harmless at the same time. The evasiveness problem largely disappears.
- **Absolute harmfulness score:** Measured on a 0–4 scale using a fine-tuned scoring model; RL-CAI becomes progressively less harmful during training while helpful RLHF gets _worse_.

---

## 6. Limitations

Honestly, a few things I think deserve more scrutiny:

- **The constitution is hand-crafted and ad hoc.** The 16 principles were selected informally. Who decides what goes in? Different stakeholders might write very different constitutions, and the paper doesn't study that variance much.
- **Self-referential feedback loop.** The same family of models writes the principles, generates the responses, and evaluates them. It's not obvious how well this generalizes or catches _subtle_ harms the model was never trained to see.
- **Goodharting is real.** They explicitly note that over-trained RL-CAI models start adding boilerplate like _"you are valid, valued, and cared for"_ to almost every response — a clear sign the model is gaming the reward.
- **Helpfulness labels still rely on humans.** The paper eliminates human labels for harmlessness, but helpfulness supervision is still fully human-dependent. It's a partial solution.
- **Evaluation is crowdworker-dependent.** Elo scores are based on crowd preferences, and the paper itself notes that worker instructions changed between this paper and prior work — which affected scores. That's a shaky foundation.

---

_Summary written as part of the 45-Day Hard research plan._
