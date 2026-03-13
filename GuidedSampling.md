# Paper Summary: GuidedSampling — Steering LLMs Towards Diverse Candidate Solutions at Inference-Time

**Authors:** Divij Handa, Mihir Parmar, Aswin RRV, Md Nayem Uddin, Hamid Palangi, Chitta Baral (Arizona State University + Google)  
**Published:** October 2025 | [arXiv:2510.03777](https://arxiv.org/abs/2510.03777)  
**Tags:** `Inference-Time Scaling` `Repeated Sampling` `Diversity` `Reasoning` `Post-Training`

---

## 1. Problem Definition

Scaling model size is hitting a wall, you need more and more data to train bigger models, and that data is running out. So the field has shifted toward **inference-time scaling**: instead of making the model bigger, you spend more compute *at test time* to get better answers.

The simplest version of this is **Repeated Sampling (RS)**: generate 50 outputs for the same question, pick the best one. It works. But there's a fundamental problem - the model was trained to produce *one correct answer*, so even when you sample 50 times, it just keeps using the same underlying approach over and over. You get 50 variations of the same idea, not 50 genuinely different strategies.

The question this paper asks: **how do you actually force the model to explore different ways of solving a problem during inference?**

---

## 2. Why Prior Methods Fail

**Repeated Sampling (RS)** is simple and cheap, but it collapses in practice. When you analyze what concepts the model uses across 100 sampled solutions, 64% of questions are solved using fewer than 3 distinct concepts, and 36% use just *one concept* for all 100 tries. The model isn't exploring, it's just running the same reasoning path with slight temperature-induced variation.

A concrete example from the paper makes this visceral: for the MATH problem "Find the maximum value of (x-y)/(x⁴+y⁴+6) over all real numbers x and y", traditional RS has 892 out of 1000 solutions using the AM-GM inequality, all leading to the wrong answer. The model is stuck.

**Tree-of-Thought (ToT)** solves the diversity problem but is computationally brutal. It requires evaluating every intermediate thought at every node expansion, costs scale with both the number of candidates *and* the depth of the tree. You get diversity, but you pay dearly for it.

What you actually want is something in between: more diverse than RS, cheaper than ToT.

---

## 3. Key Innovation

> **Decouple exploration from generation.** Instead of letting the model implicitly explore concepts while generating solutions, explicitly force it to first generate a diverse set of concepts/theorems that could solve the problem, then generate solutions conditioned on each concept separately.

This separation, explore once, generate many, is the whole idea. It's called **GuidedSampling**.

---

## 4. Core Method Explanation

GuidedSampling has two explicit phases:

### Phase 1 - Exploration: Generate Diverse Concepts

Given a question `x`, ask the model to generate concept `c₁`, the most relevant theorem or approach for solving it. Then ask again for a *different* concept `c₂`, conditioning on `c₁` already existing so the model is pushed to explore elsewhere. Repeat for `K` concepts:

```
c_k ~ p_θ(· | x, c₁, c₂, ..., c_{k-1})
```

The iterative conditioning is the key mechanism, each new concept is generated with awareness of what's already been proposed, so the model is explicitly steered away from repetition. The loop stops either at `K` concepts or when the model says "No additional concepts found."

For the same MATH problem as above, GuidedSampling's 1000 solutions use the AM-GM inequality only 77 times instead of 892,  the remaining compute gets distributed across Cauchy-Schwarz, Trivial Inequality, Chebyshev's Inequality, and others.

### Phase 2 - Generation: Produce Solutions Per Concept

For each concept `c_k`, generate `M` solutions conditioned on both the question and that specific concept:

```
s_k^(m) ~ p_θ(s | x, c_k)   for m = 1, ..., M
```

Total solutions = K × M. If your compute budget is 100 calls total, then `M = 100/K`. There's a real trade-off here: more concepts → more diversity but fewer shots per concept. The paper finds a sweet spot where increasing K helps up to a point, then performance drops as M becomes too small to develop any individual approach.

### The Full Algorithm

```
EXPLORATION PHASE:
  C = {}
  for k = 1 to K:
      c_k ~ p_θ(· | x, c_1, ..., c_{k-1})
      if c_k == "No additional concepts": break
      C = C ∪ {c_k}

GENERATION PHASE:
  S = {}
  for each concept c_k in C:
      for m = 1 to M:
          s_k^m ~ p_θ(· | x, c_k)
          S = S ∪ {s_k^m}
  return S
```

### Theoretical Bound

The paper also proves formally when GuidedSampling beats RS. The key condition is:

```
(k_min · P(C_r | x) − 1) · P_RS(y* | x) + Σ_{c ∉ C_r} π_concept(c|x) · π_base(y*|x,c) > 0
```

In plain English: GuidedSampling wins when the model has a decent probability of generating *relevant* concepts (`P(C_r | x)` is high), and those relevant concepts provide a meaningful boost to solving the problem (`k_min >> 1`). If the model generates weak or irrelevant concepts consistently, like Qwen on HumanEval, GuidedSampling can actually hurt.

### Using GuidedSampling for Post-Training

Beyond inference, they also use GuidedSampling to generate synthetic training data. Two settings:

- **FA (Final Answer only):** Discard the concept, train only on `(question, correct solution)` pairs.
- **CAA (Concept-Augmented Answer):** Train on `(question, [concepts] + solution)`, the model learns to internalize diverse reasoning strategies before committing to one.

---

## 5. Experimental Setup

- **Models:** Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct (main study); GPT-4o-mini and Phi-4-mini also tested on MATH
- **Benchmarks:** MATH (mathematical reasoning), GPQA-Diamond (graduate-level science), HumanEval (Python code generation), OlympiadBench (olympiad-level math + science)
- **Inference budget:** 100 responses per instance; pass@k reported up to k=50
- **Training data:** 10k samples from OpenMathInstruct-2; fine-tuned for 3 epochs on 4×A100 GPUs
- **Baselines:** RS, STaR, Tree-of-Thought
- **Concept extraction:** Qwen2.5-32B-Instruct used to extract core concepts from solutions for diversity analysis

**Key results:**

| Benchmark | Avg pass@50 improvement over RS |
|---|---|
| MATH | +21.8% |
| GPQA-Diamond | +11.87% |
| HumanEval | +11.28% |
| OlympiadBench | +3.08% |

For post-training, CAA setting achieves **+7.13% pass@5** over RS on average, and fine-tuned models show out-of-domain generalization, diversity learned on MATH transfers to GPQA-Diamond and HumanEval.

Diversity numbers tell the clearest story: RS produces 1.67 average concepts per instance, FA pushes it to 2.58, and CAA reaches 3.03.

---

## 6. Limitations

- **Model-dependent fragility.** GuidedSampling's gains completely depend on whether the model can generate good, diverse concepts in the exploration phase. Qwen2.5-3B on HumanEval averages only 1.13 distinct concepts, GuidedSampling actually *hurts* performance there because you're forcing solutions down a single concept path with less compute per attempt. The method breaks if concept generation breaks.

- **Compute trade-off has no clean optimum.** The K vs M trade-off is real and task-specific. There's no principled way to set K ahead of time, you need to sweep it empirically per benchmark, which defeats some of the efficiency argument.

- **Concept quality is not verified.** The exploration phase generates concepts, but there's no mechanism to check if those concepts are actually useful before spending generation budget on them. The paper acknowledges some concepts are "irrelevant" and just hopes the model recovers — which it sometimes does, but unreliably.

- **Only tested on small models (3B).** The main results are all on 3B parameter models. Whether the concept generation failure modes scale away or get worse with larger models isn't studied. The GPT-4o-mini experiment is encouraging but limited to MATH only.

- **The theoretical bound doesn't tell you much practically.** The condition in Theorem 1 requires knowing `P(C_r | x)` and `k_min`, quantities you can't measure without already running the experiment. The theory is sound but not actionable.

---

## 7. Three Research Questions I'd Explore Further

**Q1 - Can you use a verifier to filter concepts before generation?**  
The current pipeline generates K concepts and blindly generates M solutions per concept. A smarter version would have a lightweight verifier score each concept for relevance before committing generation budget to it. This would address the "Qwen on HumanEval" failure mode directly. I'd experiment with using the model itself to self-rate concept quality (probability-based or with a brief CoT evaluation) and only proceeding with concepts above some threshold.

**Q2 - Does the diversity gain from post-training transfer to harder models at larger scales?**  
The diversity numbers (1.67 → 3.03 concepts) are measured on Llama 3.2-3B. But the whole point of this is to help models that *already* explore somewhat. I'd run the same post-training experiment on a 7B or 13B model and check whether the diversity gains are additive (model was already somewhat diverse + training adds more) or redundant (larger model already explores enough that GuidedSampling data is unnecessary).

**Q3 - What happens when you apply GuidedSampling to GRPO/PPO training instead of just SFT?**  
The paper only uses GuidedSampling for supervised fine-tuning. But the diversity of reasoning trajectories it generates seems like exactly the kind of thing that would help RL training, GRPO benefits from having a wide range of attempts to compute group-relative rewards. I'd explore using GuidedSampling as the rollout strategy during RL training instead of standard repeated sampling, and measure whether the reward signal becomes more informative.

---

*Summary written as part of the 45-Day Research Sprint.*
