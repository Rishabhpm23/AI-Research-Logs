# Constitutional AI — Harmlessness from AI Feedback

**Authors:** Yuntao Bai et al. (Anthropic)  
**Published:** December 2022 | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)  
**Tags:** `AI Alignment` `RLHF` `RLAIF` `Harmlessness` `Self-Supervision`

---

<img width="1426" height="765" alt="image" src="https://github.com/user-attachments/assets/a4fec8e7-b15e-4b13-87a4-c6e2ddc28f7c" />


## 1. Problem Definition

How do you train a harmless AI assistant *without* any human labels identifying harmful outputs?

That's the exact problem this paper is going after. They want a model that's genuinely harmless — not evasive, not refusing everything — but actually harmless in a thoughtful way. And they want to do it without paying thousands of people to label what's harmful.

---

## 2. Why Prior Methods Fail

Standard RLHF for harmlessness has a core bottleneck: you need human labelers to read outputs and say "this one is more harmful than that one" — tens of thousands of times. That's expensive, slow, and the labels themselves are kind of a black box. Nobody can read 50k comparisons and understand what the model actually learned from them.

Worse, models trained this way tend to become **evasive**. The easiest way to score "harmless" with a crowd worker is to just refuse to answer. So the model learns to say "I can't help with that" — which is technically harmless but completely useless and not actually aligned.

---

## 3. Key Innovation

> Use a short list of written principles (a "constitution") as the *only* human oversight. Let the model critique and revise its own responses in a supervised phase, then replace human harmlessness labels with AI-generated ones in the RL phase. This is called **RLAIF — RL from AI Feedback**.

No human labels for harmlessness. Just rules, written in plain language.

---

## 4. Core Method Explanation

The training process has two stages:

### Supervised Learning Stage: Critique → Revision → Supervised Learning

- First, responses are generated to harmfulness prompts using a **helpful-only** AI assistant. These responses are quite toxic and harmful — that's intentional.
- Then, the model critiques its own response according to the principles in the constitution (e.g., *"Identify specific ways this response is harmful, unethical, or illegal"*).
- It then revises the original response based on that critique.
- This **critique → revision** cycle repeats multiple times, with principles drawn **randomly** from the constitution each time — so you get diversity.
- Finally, the original model is fine-tuned with supervised learning on the final revised responses.

The whole point of this stage is to get the model "on distribution" before RL kicks in — it's much easier to do RL when the model is already producing mostly reasonable outputs.

### Reinforcement Learning Stage: AI Comparison Evaluations → Preference Model → Reinforcement Learning

- This stage **mimics RLHF**, but human preferences for harmlessness are replaced with **AI Feedback (RLAIF)**.
- The model trained in the supervised stage is used to generate **pairs of responses** to each prompt in a dataset of harmful prompts.
- Each prompt + pair is formulated into a **multiple choice question**: *"Which response is better according to this constitutional principle?"* — and the AI answers it.
- This produces an **AI-generated preference dataset for harmlessness**, which is then mixed with the human feedback helpfulness dataset.
- A **preference model (PM)** is trained on this comparison data — the PM can now assign a score to any given sample.
- Finally, the model from Stage 1 is fine-tuned via **PPO (RL) against the PM**, resulting in a policy trained by RLAIF.

---

## 5. Experimental Setup

- **Models:** Series of LMs up to 52B parameters; tested at multiple scales
- **Red team prompts:** 42,496 human-written + 140,335 model-generated = ~182,831 total
- **Helpfulness prompts:** 135,296 human-written (human labels kept here)
- **Constitution:** 16 different principles, randomly sampled at each revision step
- **Evaluation:** Elo scores from crowdworker A/B comparison tests; absolute harmfulness score on a 0–4 scale
- **Key result:** RL-CAI achieves a **Pareto improvement** — more harmless *and* more helpful than standard HH RLHF at the same time. The evasiveness problem is largely solved.

---

## 6. Limitations

- The constitution is ad hoc — 16 principles chosen informally. Different stakeholders would write very different rules, and the paper doesn't test sensitivity to that.
- It's a self-referential loop: the same model family generates responses, critiques them, and evaluates them. Subtle harms the model can't perceive in itself won't get caught.
- **Goodharting is real** — over-trained RL-CAI models start adding boilerplate like "you are valid, valued, and cared for" to nearly every response. The reward is being gamed.
- Human labels for helpfulness are still required. This is a partial step toward self-supervision, not the full thing.
- Elo scores depend heavily on crowdworker instructions, which changed between this paper and prior work, making direct comparisons tricky.

---

*Summary written as part of the 45-Day Hard research plan.*
