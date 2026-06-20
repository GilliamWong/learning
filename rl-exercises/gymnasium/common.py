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
