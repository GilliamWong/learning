#!/usr/bin/env python3
"""
build_sutton_barto.py — strip the ShangtongZhang RL-an-introduction scripts into
two practice tiers (bare / with-headers), each with a descriptive top comment.

  bare/          top comment + imports only (you write everything else)
  with-headers/  top comment + imports + constants + function/method headers
                 with 'raise NotImplementedError' bodies (a first pass)

The solution clone (sutton-barto-solutions/) is never modified.
"""
from __future__ import annotations
import os
import sys
import shutil
import py_compile

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from strip_code import strip_source, bare_source   # noqa: E402

SOL = os.path.join(RL, "sutton-barto-solutions")
DST = os.path.join(RL, "sutton-barto")

# Top-of-file descriptions: what each script reproduces (problem + figures),
# its inputs/outputs/behavior — without giving away the algorithm internals.
TOP = {
 "chapter01/tic_tac_toe.py": "Tic-Tac-Toe (Chapter 1, the introductory example). Two agents learn to play by repeated self-play, updating a state-value table toward game outcomes (temporal-difference learning); a trained agent can then play a human.\nInput: none (self-play episodes). Output: a learned value table + win/draw statistics (and an interactive game).",
 "chapter02/ten_armed_testbed.py": "The 10-armed bandit testbed (Chapter 2, Figures 2.1-2.6). Implement a k-armed bandit and the action-value methods that learn on it: epsilon-greedy, optimistic initial values, UCB, and gradient bandits.\nInput: none (parameters are module constants). Output: matplotlib figures of average reward and % optimal action.",
 "chapter03/grid_world.py": "The 5x5 gridworld (Chapter 3, Figures 3.2 & 3.5). Compute the state-value function under the equiprobable random policy and the optimal value function/policy by solving the Bellman expectation and optimality equations.\nInput: none. Output: value-grid figures.",
 "chapter04/grid_world.py": "The 4x4 gridworld (Chapter 4, Figure 4.1). Iterative policy evaluation of the equiprobable random policy until convergence.\nInput: none. Output: the value function across sweeps.",
 "chapter04/car_rental.py": "Jack's car rental (Chapter 4, Figure 4.2). Policy iteration on a two-location rental MDP with Poisson demand and returns.\nInput: none. Output: the sequence of improved policies and the final value function.",
 "chapter04/car_rental_synchronous.py": "Jack's car rental (Chapter 4, Figure 4.2), synchronous-update variant. Same policy-iteration problem with synchronous value sweeps.\nInput: none. Output: improved policies and the final value function.",
 "chapter04/gamblers_problem.py": "The gambler's problem (Chapter 4, Figure 4.3). Value iteration on a coin-flip betting MDP.\nInput: none. Output: value estimates across sweeps and the final capital->stake policy.",
 "chapter05/blackjack.py": "Blackjack via Monte Carlo (Chapter 5, Figures 5.1-5.3). Implement Monte Carlo prediction (on-policy), Monte Carlo control with exploring starts, and off-policy estimation via ordinary vs weighted importance sampling.\nInput: none. Output: value-surface and optimal-policy figures.",
 "chapter05/infinite_variance.py": "The infinite-variance example (Chapter 5, Figure 5.4). Show that ordinary importance-sampling estimates can have infinite variance on a one-state problem.\nInput: none. Output: estimate trajectories across many runs.",
 "chapter06/random_walk.py": "The 5-state random walk (Chapter 6, Example 6.2 / Figure 6.2). Compare TD(0) and constant-alpha Monte Carlo value estimation, including batch updating.\nInput: none. Output: learned values and RMS-error curves.",
 "chapter06/windy_grid_world.py": "The windy gridworld (Chapter 6, Figure 6.3 / Example 6.5). On-policy TD control with Sarsa in a gridworld with upward 'wind'.\nInput: none. Output: the learning curve (time steps vs episodes).",
 "chapter06/cliff_walking.py": "The cliff-walking task (Chapter 6, Figures 6.4 & 6.6). Compare Sarsa, Expected Sarsa, and Q-learning on a gridworld with a cliff edge.\nInput: none. Output: reward-per-episode curves and the greedy policies.",
 "chapter06/maximization_bias.py": "Maximization bias (Chapter 6, Figure 6.7). Compare Q-learning and Double Q-learning on the small MDP that exposes maximization bias.\nInput: none. Output: %-left-action vs episode curves.",
 "chapter07/random_walk.py": "The 19-state random walk (Chapter 7, Figure 7.2). n-step TD prediction: study how performance varies with the number of steps n and the step size alpha.\nInput: none. Output: RMS-error vs alpha curves for several n.",
 "chapter08/maze.py": "The Dyna maze (Chapter 8, Figures 8.2/8.4/8.5, Example 8.4). Model-based RL: Dyna-Q and Dyna-Q+ with planning, plus prioritized sweeping, on gridworld mazes (including blocking/shortcut changes).\nInput: none. Output: learning curves vs number of planning steps.",
 "chapter08/expectation_vs_sample.py": "Expected vs sample updates (Chapter 8, Figure 8.7). Compare the efficiency of expected and sample updates as a function of branching factor.\nInput: none. Output: error-vs-computation curves.",
 "chapter08/trajectory_sampling.py": "Trajectory sampling (Chapter 8, Figure 8.8). Compare on-policy trajectory sampling against a uniform update distribution on random MDPs.\nInput: none. Output: value-of-start-state vs computation.",
 "chapter09/random_walk.py": "The 1000-state random walk (Chapter 9, Figures 9.1/9.2/9.5/9.10). On-policy prediction with function approximation: gradient Monte Carlo and semi-gradient n-step TD using state aggregation, polynomial/Fourier bases, and tile coding.\nInput: none. Output: approximate value functions and error curves.",
 "chapter09/square_wave.py": "The square-wave coarse-coding example (Chapter 9, Figure 9.8). Show how feature width affects generalization and asymptotic accuracy when approximating a square wave.\nInput: none. Output: learned approximations for several feature widths.",
 "chapter10/mountain_car.py": "Mountain Car (Chapter 10, Figures 10.1-10.4). On-policy control with function approximation: episodic semi-gradient Sarsa with tile coding (and n-step variants).\nInput: none. Output: cost-to-go surfaces and learning curves.",
 "chapter10/access_control.py": "The access-control queuing task (Chapter 10, Figure 10.5). Average-reward control with differential semi-gradient Sarsa.\nInput: none. Output: the learned value/policy over free servers and priorities.",
 "chapter11/counterexample.py": "Baird's counterexample (Chapter 11, Figures 11.2/11.6/11.7). Demonstrate off-policy divergence of semi-gradient TD, and the stabler behavior of TDC/Gradient-TD and Emphatic-TD.\nInput: none. Output: weight-trajectory curves.",
 "chapter12/random_walk.py": "The 19-state random walk with eligibility traces (Chapter 12, Figures 12.3/12.6/12.8). Implement the offline lambda-return algorithm, TD(lambda), and true online TD(lambda).\nInput: none. Output: RMS-error vs (alpha, lambda) curves.",
 "chapter12/mountain_car.py": "Mountain Car with eligibility traces (Chapter 12, Figures 12.10 & 12.11). Sarsa(lambda) with replacing traces and tile coding.\nInput: none. Output: learning curves comparing trace variants.",
 "chapter12/lambda_effect.py": "The effect of the trace-decay parameter lambda (Chapter 12). Sweep lambda across the chapter's tasks/algorithms to show its influence on performance.\nInput: none. Output: performance-vs-lambda curves.",
 "chapter13/short_corridor.py": "The short-corridor gridworld (Chapter 13, Figures 13.1 & 13.2). Policy-gradient control with REINFORCE (and REINFORCE with baseline) on a small problem whose optimal policy is stochastic.\nInput: none. Output: total-reward-vs-episode curves.",
}


def make_top_comment(rel):
    body = TOP.get(rel, f"Sutton & Barto exercise script: {rel}")
    fname = os.path.basename(rel)
    lines = [f"# {fname} -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)", "#"]
    for para in body.split("\n"):
        # wrap long lines at ~78 chars on word boundaries
        words, cur = para.split(" "), ""
        for w in words:
            if len(cur) + len(w) + 1 > 76:
                lines.append("# " + cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append("# " + cur)
        lines.append("#")
    while lines and lines[-1] == "#":
        lines.pop()
    return "\n".join(lines)


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    py_files = []
    for root, _, files in os.walk(SOL):
        if ".git" in root:
            continue
        for fn in sorted(files):
            if fn.endswith(".py"):
                py_files.append(os.path.relpath(os.path.join(root, fn), SOL))
    py_files.sort()

    for rel in py_files:
        with open(os.path.join(SOL, rel), encoding="utf-8") as f:
            src = f.read()
        top = make_top_comment(rel)
        for tier in ("bare", "with-headers"):
            outp = os.path.join(DST, tier, rel)
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            if tier == "bare":
                out = bare_source(src, top)
            else:
                out = top.rstrip() + "\n\n\n" + strip_source(src).lstrip("\n")
            with open(outp, "w", encoding="utf-8") as f:
                f.write(out)
        print(f"  {rel:42}  stripped ({strip_source(src).count('raise NotImplementedError')} stubs)")

    # compile-check
    bad = 0
    for root, _, files in os.walk(DST):
        for fn in files:
            if fn.endswith(".py"):
                try:
                    py_compile.compile(os.path.join(root, fn), doraise=True)
                except py_compile.PyCompileError as e:
                    bad += 1
                    print(f"  !! {fn}: {e}")
    print(f"\nFiles: {len(py_files)} x 2 tiers. Compile check: {'PASSED' if bad == 0 else f'FAILED ({bad})'}")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
