#!/usr/bin/env python3
"""
gen_gym_scaffolds.py — generate a policy-writing scaffold for every Gymnasium
environment that can be created in this install.

For each makeable env it writes  gymnasium/<family>/<EnvId>.py  containing the
env's official documentation (spaces, rewards, termination) + a runnable Policy
skeleton you fill in. A shared gymnasium/common.py provides the roll-out loop.
Envs that need uninstalled extras (e.g. MuJoCo) are skipped and reported so you
know what `pip install gymnasium[...]` would unlock; re-run this to add them.
"""
from __future__ import annotations
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
OUT = os.path.join(RL, "gymnasium")

import gymnasium as gym  # noqa: E402

SKIP_IDS = {"GymV21Environment-v0", "GymV26Environment-v0"}

COMMON = '''\
"""Shared helpers for the Gymnasium policy exercises.

Write a Policy in any env file (subclass BasePolicy, override act()/update()),
then run that file. `run()` rolls episodes, prints returns, and works with any
observation/action space.
"""
from __future__ import annotations
import argparse
import statistics
import gymnasium as gym


class BasePolicy:
    """Base class for your policies.

    observation_space / action_space are the env's gym.spaces, so you can size
    tables/networks and (by default) sample random actions.
    """

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """Called at the start of each episode. Override if your policy has
        per-episode state (eligibility traces, an RNN hidden state, ...)."""
        pass

    def act(self, observation):
        """Return an action for `observation`. Override this — the default is a
        random action so a fresh scaffold runs out of the box."""
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        """Called after every step with the transition. Override to learn
        online (Q-learning, Sarsa, policy gradient, ...). No-op by default."""
        pass


def run(env_id, policy_cls=BasePolicy, episodes=5, render=False, seed=None,
        max_steps=None, **make_kwargs):
    env = gym.make(env_id, render_mode="human" if render else None, **make_kwargs)
    policy = policy_cls(env.observation_space, env.action_space)
    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=(seed + ep) if seed is not None else None)
        policy.reset()
        done, total, steps = False, 0.0, 0
        while not done:
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy.update(obs, action, reward, next_obs, terminated, truncated, info)
            obs, total, steps = next_obs, total + reward, steps + 1
            done = terminated or truncated or (max_steps is not None and steps >= max_steps)
        returns.append(total)
        print(f"episode {ep + 1}/{episodes}: return={total:.3f} steps={steps}")
    env.close()
    if returns:
        print(f"mean return over {len(returns)} eps: {statistics.fmean(returns):.3f}")
    return returns


def parse_args(default_episodes=5):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=default_episodes)
    p.add_argument("--render", action="store_true", help="watch the agent (human render)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()
'''


def family_of(spec):
    ep = spec.entry_point
    if isinstance(ep, str) and ep.startswith("gymnasium.envs."):
        return ep[len("gymnasium.envs."):].split(".")[0]
    return "other"


def trim_doc(doc):
    if not doc:
        return ["(no built-in description — see https://gymnasium.farama.org/)"]
    out, stop = [], ("## Arguments", "## Version History", "## References", "## Credits")
    for line in doc.splitlines():
        if line.strip().startswith(stop):
            break
        out.append(line.rstrip())
    # collapse leading/trailing blanks, cap length
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out[:70] if out else ["(no description)"]


def scaffold(env_id, family, obs_space, act_space, max_steps, doc_lines):
    safe = env_id.replace("/", "_").replace("-", "_")
    discrete = isinstance(act_space, gym.spaces.Discrete)
    if discrete:
        act_hint = f"# return an integer action in [0, {act_space.n})"
    elif isinstance(act_space, gym.spaces.Box):
        act_hint = f"# return a numpy array, shape {tuple(act_space.shape)}, within the action bounds"
    else:
        act_hint = "# return an action that is a member of self.action_space"
    header = "\n".join("# " + l if l.strip() else "#" for l in doc_lines)
    obs_s = " ".join(str(obs_space).split())  # collapse multi-line numpy reprs
    act_s = " ".join(str(act_space).split())
    return safe, f'''\
# {"=" * 74}
# {env_id}   (family: {family})
# {"=" * 74}
{header}
#
# ---- quick reference ----------------------------------------------------
# Observation space: {obs_s}
# Action space:      {act_s}
# Max episode steps: {max_steps}
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python {os.path.basename(safe)}.py --episodes 20            # evaluate
#   python {os.path.basename(safe)}.py --render                 # watch it play
# {"=" * 74}
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "{env_id}"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        {act_hint}
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
'''


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "common.py"), "w", encoding="utf-8") as f:
        f.write(COMMON)

    generated, skipped = [], []
    for env_id, spec in sorted(gym.envs.registry.items()):
        if env_id in SKIP_IDS:
            continue
        try:
            env = gym.make(env_id, disable_env_checker=True)
            obs_space, act_space = env.observation_space, env.action_space
            doc = env.unwrapped.__doc__
            max_steps = getattr(spec, "max_episode_steps", None)
            env.close()
        except Exception as e:  # noqa: BLE001
            msg = textwrap.shorten(" ".join(str(e).split()), width=200, placeholder=" ...")
            skipped.append((env_id, f"{type(e).__name__}: {msg}"))
            continue
        fam = family_of(spec)
        safe, code = scaffold(env_id, fam, obs_space, act_space, max_steps, trim_doc(doc))
        d = os.path.join(OUT, fam)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, safe + ".py"), "w", encoding="utf-8") as f:
            f.write(code)
        generated.append((fam, env_id, safe))

    # index README
    by_fam = {}
    for fam, env_id, safe in generated:
        by_fam.setdefault(fam, []).append((env_id, safe))
    lines = ["# Gymnasium policy scaffolds", "",
             f"{len(generated)} environments scaffolded across {len(by_fam)} families.",
             "Each file is a runnable random-policy template — implement `Policy.act()`.", ""]
    for fam in sorted(by_fam):
        lines.append(f"### {fam}/")
        for env_id, safe in sorted(by_fam[fam]):
            lines.append(f"- `{fam}/{safe}.py` — `{env_id}`")
        lines.append("")
    if skipped:
        lines.append("### Not generated (need extra installs)")
        lines.append("These are registered but couldn't be created in this environment "
                     "(e.g. MuJoCo needs `pip install gymnasium[mujoco]`, Atari needs "
                     "`pip install gymnasium[atari] ale-py`). Re-run "
                     "`python _tools/gen_gym_scaffolds.py` after installing to add them:")
        lines.append("")
        for env_id, why in skipped:
            lines.append(f"- `{env_id}` — {why}")
        lines.append("")
    with open(os.path.join(OUT, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    import py_compile
    bad = 0
    for fam, _, safe in generated:
        try:
            py_compile.compile(os.path.join(OUT, fam, safe + ".py"), doraise=True)
        except py_compile.PyCompileError as e:
            bad += 1
            print(f"  !! compile FAIL {safe}: {e}")
    print(f"compile check: {'PASSED' if bad == 0 else f'FAILED ({bad})'}")
    print(f"generated {len(generated)} scaffolds in {len(by_fam)} families: "
          + ", ".join(f"{k}({len(v)})" for k, v in sorted(by_fam.items())))
    print(f"skipped {len(skipped)} (need extra installs): "
          + ", ".join(sorted({why.split(':')[0] for _, why in skipped})) if skipped else "skipped 0")
    return generated, skipped


if __name__ == "__main__":
    main()
