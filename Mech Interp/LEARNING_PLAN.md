# Your next steps

**Start with ARENA 1.1: Transformers from Scratch.** Your saved implementations of all seven opening exercises pass their checks. You have also skimmed the opening part of Neel's guide, so there is no need to repeat that orientation.

Use this plan as your main sequence. The [ARENA catalog](arena/README.md) contains every imported chapter, including later branches. Study the exercise copies; the reference solutions are in a separate folder.

## Before or alongside your next session — close the warm-up notes

Budget roughly 20–40 minutes across these items. They need not prevent you from starting the transformer material.

- In Module 1, finish **Explain and vary**: describe one bug or surprising result, and one input variation you checked. Your initial broadcasting explanation is saved.
- In Module 2, record one controlled learning-rate comparison and write a **claim, evidence, limitation, and next experiment**. Some of these written fields are still blank.
- If you already ran an experiment, record it as an observation. Make a fresh prediction for a new variation rather than reconstructing a prediction after seeing the answer.
- Save the notebooks, then ask for a review here. The code checks establish the tested behavior; the explanations are reviewed separately.

## 1. Understand transformer inputs and outputs

**Work:** [ARENA 1.1 exercise notebook](arena/notebooks/chapter1_transformer_interp/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb), **Understanding Inputs & Outputs of a Transformer**.

**Read:** the **Transformer Overview** in [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), particularly the high-level architecture and residual stream. Revisit the transformer-basics subsection of [Neel's researcher guide](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher) only where it answers a question you have.

**Write:** trace a short prompt from token IDs to the scores used to predict its next token. Label batch, position, feature, and vocabulary dimensions.

**Move on when:** you can explain what each position's output predicts and why causal masking is needed.

## 2. Implement the transformer

**Work:** the same ARENA 1.1 notebook, **Clean Transformer Implementation**. Follow its component order and run each supplied check before assembling the full model.

**Read as needed:** Neel's [transformer coding walkthrough](https://www.youtube.com/watch?v=bOYE6E8JrtU&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz), using the portion about the component you are implementing. Use [his glossary](https://neelnanda.io/glossary) for unfamiliar terms.

**Write:** a dimension trace through attention and one residual block. Explain what changes token positions and what operates independently at each position.

**Move on when:** the component checks pass, the assembled model loads the provided weights, and its outputs match the reference. Documentation and hints are allowed throughout.

## 3. Train and sample once

**Work:** ARENA 1.1, **Training a Transformer**, then the initial **Sampling** material. Run a short training example first; you can lower the training argument values while debugging. The original longer training settings remain available.

**Write:** what the loss measures, what improvement you observed, and what the generated samples do or do not show. Compare one sampling setting while keeping the model fixed.

**Defer initially:** advanced beam-search and KV-cache work if it is delaying your first interpretability experiment. Return when those mechanics become useful.

## 4. Inspect an induction mechanism

**Work:** [ARENA 1.2 exercise notebook](arena/notebooks/chapter1_transformer_interp/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb), sections **1–3**: loading/caching, finding induction heads, and hooks.

**Read:** the introduction and induction-head definition in the [local induction paper reader](library/readers/Induction_Heads.ipynb). Before investigating ablation, read **Argument 3: Direct ablation** and the relevant method in the [web edition](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).

**Write before looking:** the attention pattern you expect on a repeated sequence. Afterward, explain any mismatch.

**Move on when:** you can inspect activations, identify a candidate head, and perform a controlled intervention using hooks.

## 5. Do your first small extension

**Read first:** **Stage 2: Practicing Research with Mini-Projects** in Neel's guide, especially exploration and hypothesis testing. You do not need to wait until the end of the course to use this advice.

**Experiment:** extend one induction exercise. A concrete option is to vary the distance between repeated tokens and measure the effect of ablating a candidate head.

Before running, specify the prediction, metric, data construction, what stays fixed, and a control. Use several examples and reserve fresh examples for checking the conclusion. Record whether an effect is specific to the behavior or reflects broader damage to the model.

**Deliverable:** one runnable [experiment notebook](templates/Experiment.ipynb), one figure, and a short paragraph separating observation, inference, and uncertainty. An inconclusive or negative result is still a valid outcome.

## 6. Learn activation patching through IOI

**Work:** [ARENA IOI exercises](arena/notebooks/chapter1_transformer_interp/part41_indirect_object_identification/1.4.1_Indirect_Object_Identification_exercises.ipynb), sections **1–3**: task setup, logit attribution, and activation patching.

**Read:** the task definition, metric, and chosen patching method in the [local IOI paper reader](library/readers/Indirect_Object_Identification.ipynb). Read the figure or result you are targeting closely enough to specify its setup.

**Deliverable:** reproduce one identified result, including the baseline and control. Record all differences from the source setup. Treat an adapted experiment as an adaptation.

**Optional deepening:** sections 4–5 for path patching and a fuller circuit replication. Choose this if circuit analysis is the direction you want to explore next.

## 7. Broaden into representations

After the first small experiment and IOI introduction, use this order:

1. [Toy Models of Superposition & SAEs](arena/notebooks/chapter1_transformer_interp/part54_toy_models_of_superposition_and_saes/1.5.4_Toy_Models_of_Superposition_%26_SAEs_exercises.ipynb): start with the toy model and synthetic-data experiments. Pair it with the toy setup in the [local paper reader](library/readers/Toy_Models_of_Superposition.ipynb).
2. [Linear Probes](arena/notebooks/chapter1_transformer_interp/part31_linear_probes/1.3.1_Linear_Probes_exercises.ipynb): learn what a probe can establish about represented information. Check the gated-model and memory requirements before the larger-model sections.
3. [Interpretability with SAEs](arena/notebooks/chapter1_transformer_interp/part33_interp_with_saes/1.3.3_Interpretability_with_SAEs_exercises.ipynb): study use of pretrained SAEs and their limitations. Advanced sections need model access and some use paid APIs.
4. Read Goodfire's [Understanding and Steering Llama 3](https://www.goodfire.com/research/understanding-and-steering-llama-3) as an applied comparison. The article's old demo/API was discontinued; the reading itself is still useful.

Then choose between model steering, circuit tracing, grokking, OthelloGPT, or activation oracles based on the question you want to investigate. You do not need to finish every branch before attempting another project.

## 8. Start a longer investigation

Read **Stage 3** of Neel's guide when you begin a longer project. Define a narrow question, the prior result it builds on, a small pilot, and a decision point. Use the [experiment template](templates/Experiment.ipynb) and [paper-note template](templates/Paper_Notes.ipynb).

Success means a reproducible result you can explain, including its limits. Check the literature before claiming novelty.

## Other imported chapters — when to use them

These are available in the catalog. They are useful branches, not prerequisites for your first mech-interp project.

| Branch | Concrete order | When to choose it |
|---|---|---|
| Extra fundamentals | 0.0 tensor practice → 0.1 ray tracing or 0.2 CNNs → 0.3 optimization → 0.4 backprop → 0.5 generative models | Fill a specific coding or neural-network gap. Your initial coding checks already pass. |
| Reinforcement learning | 2.1 → 2.2.1 DQN → 2.2.2 VPG → 2.3 PPO → 2.4 RLHF; 2.5 MCTS/AlphaZero afterward or as a separate interest | You want to study training, agents, or behavior shaped by RL. Linux-only and multi-GPU bonuses are deferred. |
| LLM evaluations | 3.1 → 3.2 → 3.3 → 3.4 → 3.5 | Your research question needs behavioral measurements, evaluation datasets, or agent experiments. Paid calls are manual. |
| Alignment science | 4.1 → 4.2 → 4.3 → 4.4 → 4.5 | After you can run basic model experiments and evaluations. These case studies have additional API/model-access requirements. |

## A working rhythm

At the beginning of a session, choose one specific task. Read the relevant section, predict something, implement or test it, then write what changed in your understanding. End by saving a concrete next action. Ask for a review of saved explanations or results whenever you want feedback.
