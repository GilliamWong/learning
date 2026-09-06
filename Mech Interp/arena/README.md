# ARENA notebooks

**Your next notebook is [1.1 · Transformer from Scratch](notebooks/chapter1_transformer_interp/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb).**
Use [your study plan](../LEARNING_PLAN.md) for the reading, coding, and experiment order.

## Open and work

1. Open an exercise notebook below. The appropriate ARENA kernel is selected automatically.
2. Run the local setup cell first, then work through the notebook with **Shift+Enter**.
3. Implement each exercise before running its tests. Blank answers and `NotImplementedError` are intentional.
4. Save your code and writing with **Ctrl+S**. Use the workbench module checklists for your milestones.

The environment is installed. [Check the GPU and plots](00_Environment_Check.ipynb) if something looks wrong.
Most notebooks use **ARENA (local GPU)**. Notebook 1.4.2 uses **ARENA (SAE circuits)** because its package
versions differ from the older exercises. You do not need to switch versions manually.
Use **Setup ARENA.cmd** in the workspace folder only if you need to rebuild it. Restart a kernel when switching
between chapters, because different chapters have helpers with the same names. Close unused kernels to free GPU memory.

## What has been checked

Imported **75 notebooks: 34 exercises, 34 references, and 7 additional training notebooks**, covering all five chapters.
**75/75 setup and import checks passed.** Backprop, Transformer from Scratch, Intro to Mech Interp, and IOI
reference notebooks also passed complete sequential runs. More numerical checks are recorded in [validation details](VALIDATION.md).

Full execution has not been verified for every advanced notebook. Large models, gated checkpoints, long training runs,
and paid API sections have extra requirements. Paid calls stay manual, and Linux-only/multi-GPU extensions are deferred
by your choice. The notes below identify these boundaries before you start a branch.

References are in a [separate solutions index](reference/README.md), and the seven extra notebooks have an
[additional training index](additional/README.md).


## 0 · Fundamentals

| Exercise notebook | Current scope |
|---|---|
| [0.0 Prerequisites](notebooks/chapter0_fundamentals/part0_prereqs/0.0_Prerequisites_exercises.ipynb) | Local numerical checks passed; full reference run not yet checked. |
| [0.1 Ray Tracing](notebooks/chapter0_fundamentals/part1_ray_tracing/0.1_Ray_Tracing_exercises.ipynb) | Local numerical checks passed; full reference run not yet checked. |
| [0.2 CNNs & ResNets](notebooks/chapter0_fundamentals/part2_cnns/0.2_CNNs_%26_ResNets_exercises.ipynb) | Local numerical checks passed; full reference run not yet checked. |
| [0.3 Optimization](notebooks/chapter0_fundamentals/part3_optimization/0.3_Optimization_exercises.ipynb) | Core optimizer tests passed. Distributed/NCCL extras deferred. |
| [0.4 Backprop](notebooks/chapter0_fundamentals/part4_backprop/0.4_Backprop_exercises.ipynb) | Full sequential reference run passed. |
| [0.5 VAEs & GANs](notebooks/chapter0_fundamentals/part5_vaes_and_gans/0.5_VAEs_%26_GANs_exercises.ipynb) | 12 component tests passed; dataset downloads and full training remain to run. |

## 1 · Transformer interpretability

| Exercise notebook | Current scope |
|---|---|
| [1.1 Transformer from Scratch](notebooks/chapter1_transformer_interp/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb) | Start here. Full reference run passed, including training and sampling. |
| [1.2 Intro to Mech Interp](notebooks/chapter1_transformer_interp/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb) | Full reference run passed. Follow after 1.1. |
| [1.3.1 Linear Probes](notebooks/chapter1_transformer_interp/part31_linear_probes/1.3.1_Linear_Probes_exercises.ipynb) | Later branch. Gated models and substantial memory requirements. |
| [1.3.2 Function Vectors & Model Steering](notebooks/chapter1_transformer_interp/part32_function_vectors_and_model_steering/1.3.2_Function_Vectors_%26_Model_Steering_exercises.ipynb) | Later branch. Model downloads and optional remote/API execution. |
| [1.3.3 Interpretability with SAEs](notebooks/chapter1_transformer_interp/part33_interp_with_saes/1.3.3_Interpretability_with_SAEs_exercises.ipynb) | Later branch. Pretrained SAEs, model access, and optional paid APIs. |
| [1.3.4 Activation Oracles](notebooks/chapter1_transformer_interp/part34_activation_oracles/1.3.4_Activation_Oracles_exercises.ipynb) | Later branch. Oracle checkpoints and larger model requirements. |
| [1.4.1 Indirect Object Identification](notebooks/chapter1_transformer_interp/part41_indirect_object_identification/1.4.1_Indirect_Object_Identification_exercises.ipynb) | Full reference run passed. Follow induction and a small extension. |
| [1.4.2 SAE Circuits](notebooks/chapter1_transformer_interp/part42_sae_circuits/1.4.2_SAE_Circuits_exercises.ipynb) | Separate SAE circuits kernel prepared and checked. Gated model/SAE artifacts; full execution remains manual. |
| [1.5.1 Balanced Bracket Classifier](notebooks/chapter1_transformer_interp/part51_balanced_bracket_classifier/1.5.1_Balanced_Bracket_Classifier_exercises.ipynb) | Later small-model circuit investigation; full reference run not yet checked. |
| [1.5.2 Grokking & Modular Arithmetic](notebooks/chapter1_transformer_interp/part52_grokking_and_modular_arithmetic/1.5.2_Grokking_%26_Modular_Arithmetic_exercises.ipynb) | Later branch. Numerical helpers checked; checkpoints/training remain to run. |
| [1.5.3 OthelloGPT](notebooks/chapter1_transformer_interp/part53_othellogpt/1.5.3_OthelloGPT_exercises.ipynb) | Later branch. Model/data downloads; full reference run not yet checked. |
| [1.5.4 Toy Models of Superposition & SAEs](notebooks/chapter1_transformer_interp/part54_toy_models_of_superposition_and_saes/1.5.4_Toy_Models_of_Superposition_%26_SAEs_exercises.ipynb) | Recommended representations branch. 12 numerical checks passed; full training not yet checked. |

## 2 · Reinforcement learning

| Exercise notebook | Current scope |
|---|---|
| [2.1 Intro to RL](notebooks/chapter2_rl/part1_intro_to_rl/2.1_Intro_to_RL_exercises.ipynb) | Local numerical checks passed; full reference run not yet checked. |
| [2.2.1 DQN](notebooks/chapter2_rl/part21_dqn/2.2.1_DQN_exercises.ipynb) | Local numerical checks passed; full reference run not yet checked. |
| [2.2.2 VPG](notebooks/chapter2_rl/part22_vpg/2.2.2_VPG_exercises.ipynb) | Local numerical checks passed; full reference run not yet checked. |
| [2.3 PPO](notebooks/chapter2_rl/part3_ppo/2.3_PPO_exercises.ipynb) | Core PPO numerical checks passed. EnvPool and JAX/Brax extensions deferred. |
| [2.4 RLHF](notebooks/chapter2_rl/part4_rlhf/2.4_RLHF_exercises.ipynb) | 10 numerical checks passed. Full RLHF training and model fixtures not yet checked. |
| [2.5 MCTS & AlphaZero](notebooks/chapter2_rl/part5_mcts_alphazero/2.5_MCTS_%26_AlphaZero_exercises.ipynb) | 32 component checks passed, including self-play/training steps. Full AlphaZero training not run. |

## 3 · LLM evaluations

| Exercise notebook | Current scope |
|---|---|
| [3.1 Intro to Evals](notebooks/chapter3_llm_evals/part1_intro_to_evals/3.1_Intro_to_Evals_exercises.ipynb) | Paid model calls are manual. |
| [3.2 Dataset Generation](notebooks/chapter3_llm_evals/part2_dataset_generation/3.2_Dataset_Generation_exercises.ipynb) | Paid data-generation and evaluation calls are manual. |
| [3.3 Running Evals with Inspect](notebooks/chapter3_llm_evals/part3_running_evals_with_inspect/3.3_Running_Evals_with_Inspect_exercises.ipynb) | Inspect setup checked; live evaluations are manual. |
| [3.4 LLM Agents](notebooks/chapter3_llm_evals/part4_llm_agents/3.4_LLM_Agents_exercises.ipynb) | Local helper checks passed; model/agent calls are manual. |
| [3.5 AI Control](notebooks/chapter3_llm_evals/part5_ai_control/3.5_AI_Control_exercises.ipynb) | 11 local checks passed. Live agents and Docker sandbox experiments remain manual. |

## 4 · Alignment science

| Exercise notebook | Current scope |
|---|---|
| [4.1 Emergent Misalignment](notebooks/chapter4_alignment_science/part1_emergent_misalignment/4.1_Emergent_Misalignment_exercises.ipynb) | Manual API/model access; training and evaluation not run. |
| [4.2 Science of Misalignment](notebooks/chapter4_alignment_science/part2_science_of_misalignment/4.2_Science_of_Misalignment_exercises.ipynb) | Manual API/model access; large model experiments not run. |
| [4.3 Interpreting Reasoning Models](notebooks/chapter4_alignment_science/part3_interpreting_reasoning_models/4.3_Interpreting_Reasoning_Models_exercises.ipynb) | Manual API/model access; reasoning-model experiments not run. |
| [4.4 LLM Psychology & Persona Vectors](notebooks/chapter4_alignment_science/part4_persona_vectors/4.4_LLM_Psychology_%26_Persona_Vectors_exercises.ipynb) | Manual API/model access; large model experiments not run. |
| [4.5 Investigator Agents](notebooks/chapter4_alignment_science/part5_investigator_agents/4.5_Investigator_Agents_exercises.ipynb) | Compatible Petri imports checked. Live API investigations remain manual. |

## APIs and advanced sections

No paid calls were made during setup or validation. When you choose to run an API section, copy `.env.example`
to `.env` in this `arena` folder and fill only the keys that section needs. Keep that file private; it is ignored
by Git. Restart the notebook kernel and rerun setup. Choose model, request count, and spending limits before
starting a live evaluation. A notebook cell can make many requests, so work through API notebooks cell by cell.

This Windows environment includes core RL, Gymnasium, and MuJoCo. The course's optional EnvPool and CUDA
JAX/Brax sections need a later Linux environment; NCCL/distributed GPU exercises also need appropriate hardware.
Some later high-throughput model tools, including vLLM-based assistant-axis workflows, also need Linux.
Docker-based agent sandboxes need Docker configured before use. The fixed local sandbox test was checked with
trusted test code; running model-generated code belongs in the intended isolated sandbox.

Supporting repositories have been fetched, but large Git LFS artifacts are not downloaded automatically.
Gated Hugging Face models need your own access approval. The installed GPU has 16 GB VRAM, so some larger-model
sections will need a different model configuration or more compute. Those sections are prepared, not certified
to fit this machine unchanged.

## Source and local changes


Source: [official ARENA snapshot `527f9376b40a`](https://github.com/callummcdougall/ARENA_3.0/tree/527f9376b40ad9a12ecd80490884b0009b54dd55).

Original notebooks remain in `ARENA_3.0`. Prepared copies replace Colab installers and paths, use Windows-compatible DataLoader defaults, and repair malformed placeholders. Exercise answers remain blank.

The import manifest records source links, hashes, and adaptations. Small support-code repairs are recorded in `support-patches.json`; fetched dependencies in `supporting-repositories.json`. Source content and attribution remain with the snapshot.
