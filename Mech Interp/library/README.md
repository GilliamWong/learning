# Reading library

Use the home page for filtering and saved reading status. This is the plain-text source index.

## ARENA 0.0 — Tensor manipulation
ARENA / Callum McDougall · Exercises · Module 01

[ARENA 0.0 — Tensor manipulation](https://learn.arena.education/chapter0_fundamentals/00_prereqs/2-einops-einsum-tensor-manipulation/)

Read Broadcasting. Use the einops and einsum examples as references when an exercise needs them.

The local exercises are original practice tasks on the same skills; this link provides the official, broader exercise set.

## Learn the Basics
PyTorch · Tutorial · Module 02

[Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

Tensors → Build Model → Autograd → Optimization. Read the relevant section when implementing that component.

Use documentation while coding. API memorization is not a completion requirement.

## ARENA 1.1 — Transformers from scratch
ARENA / Callum McDougall · Exercises · Module 03

[ARENA 1.1 — Transformers from scratch](https://learn.arena.education/chapter1_transformer_interp/01_transformers/)

Inputs & Outputs, then Clean Transformer Implementation. Read the basic training loop; defer advanced sampling and caching.

Open the page's exercises Colab, not its solutions notebook. The official notebook supplies its own environment setup.

## A Mathematical Framework for Transformer Circuits
Elhage et al. / Anthropic · Research article · Module 03

[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

First pass: Transformer Overview, especially High-Level Architecture and the Residual Stream. Revisit QK/OV and two-layer composition in Module 4.

Treat the simplifying assumptions as part of the model being studied, not properties of every transformer.

## How To Become A Mechanistic Interpretability Researcher
Neel Nanda · Research guidance · Module 03

[How To Become A Mechanistic Interpretability Researcher](https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)

Stage 1 now; mini-project and hypothesis-testing advice when starting Module 6.

The one-month guideline means about 200 working hours in the post's footnote. It is not a part-time calendar deadline.

## ARENA 1.2 — TransformerLens & induction circuits
ARENA / Callum McDougall · Exercises · Module 04

[ARENA 1.2 — TransformerLens & induction circuits](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/)

Sections 1–3 first: loading and caching, finding induction heads, and hooks. Section 4 is a later deep dive.

The guide and original exercises are linked from the local module. Transformer dependencies are not part of the opening CPU environment.

## In-context Learning and Induction Heads
Olsson et al. / Anthropic · Paper · Module 04

[In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)

[Read locally](readers/Induction_Heads.ipynb)

Preview the introduction and definition of induction heads. For the intervention exercise, study Argument 3 (Direct ablation) and the relevant methods.

Distinguish evidence in small attention-only models from broader hypotheses about in-context learning.

## ARENA 1.4.1 — Indirect Object Identification
ARENA / Callum McDougall · Exercises · Module 05

[ARENA 1.4.1 — Indirect Object Identification](https://learn.arena.education/chapter1_transformer_interp/21_ioi/)

Sections 1–3 for task setup, attribution, and activation patching. Sections 4–5 deepen the replication.

Record which part you reproduced. Completing the introductory exercise is not a full paper replication.

## Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small
Wang et al. · Paper · Module 05

[Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)

[Read locally](readers/Indirect_Object_Identification.ipynb)

Read the task definition, the metric, and the method behind the result you select. Then inspect that result's controls and limitations.

Choose and record the exact figure or claim before implementing the replication.

## Toy Models of Superposition
Elhage et al. / Anthropic · Paper · Module later

[Toy Models of Superposition](https://arxiv.org/abs/2209.10652)

[Read locally](readers/Toy_Models_of_Superposition.ipynb)

Introduction and toy setup first. Select one small configuration to reproduce if you choose the representations branch.

A small, controlled setting for investigating what a representation stores.

## Understanding and Steering Llama 3 with Sparse Autoencoders
Goodfire · Research article · Module later

[Understanding and Steering Llama 3 with Sparse Autoencoders](https://www.goodfire.com/research/understanding-and-steering-llama-3)

Read the sparse-autoencoder introduction and one steering example after studying superposition.

Read the article as an applied example. Its old demo and API were deprecated in February 2026; this assignment does not use them.

## Under the Hood of a Reasoning Model
Goodfire · Research article · Module later

[Under the Hood of a Reasoning Model](https://www.goodfire.com/research/under-the-hood-of-a-reasoning-model)

Preview the main examples, then choose one claim and ask what evidence would distinguish alternatives.

A reading assignment, not a promise that a large-model replication fits the opening environment.
