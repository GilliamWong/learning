# ==========================================================================
# Acrobot-v1   (family: classic_control)
# ==========================================================================
#     ## Description
#
#     The Acrobot environment is based on Sutton's work in
#     ["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
#     and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
#     The system consists of two links connected linearly to form a chain, with one end of
#     the chain fixed. The joint between the two links is actuated. The goal is to apply
#     torques on the actuated joint to swing the free end of the linear chain above a
#     given height while starting from the initial state of hanging downwards.
#
#     As seen in the **Gif**: two blue links connected by two green joints. The joint in
#     between the two links is actuated. The goal is to swing the free end of the outer-link
#     to reach the target height (black horizontal line above system) by applying torque on
#     the actuator.
#
#     ## Action Space
#
#     The action is discrete, deterministic, and represents the torque applied on the actuated
#     joint between the two links.
#
#     | Num | Action                                | Unit         |
#     |-----|---------------------------------------|--------------|
#     | 0   | apply -1 torque to the actuated joint | torque (N m) |
#     | 1   | apply 0 torque to the actuated joint  | torque (N m) |
#     | 2   | apply 1 torque to the actuated joint  | torque (N m) |
#
#     ## Observation Space
#
#     The observation is a `ndarray` with shape `(6,)` that provides information about the
#     two rotational joint angles as well as their angular velocities:
#
#     | Num | Observation                  | Min                 | Max               |
#     |-----|------------------------------|---------------------|-------------------|
#     | 0   | Cosine of `theta1`           | -1                  | 1                 |
#     | 1   | Sine of `theta1`             | -1                  | 1                 |
#     | 2   | Cosine of `theta2`           | -1                  | 1                 |
#     | 3   | Sine of `theta2`             | -1                  | 1                 |
#     | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
#     | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |
#
#     where
#     - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
#     downwards.
#     - `theta2` is ***relative to the angle of the first link.***
#         An angle of 0 corresponds to having the same angle between the two links.
#
#     The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
#     A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.
#
#     ## Rewards
#
#     The goal is to have the free end reach a designated target height in as few steps as possible,
#     and as such all steps that do not reach the goal incur a reward of -1.
#     Achieving the target height results in termination with a reward of 0. The reward threshold is -100.
#
#     ## Starting State
#
#     Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
#     uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.
#
#     ## Episode End
#
#     The episode ends if one of the following occurs:
#     1. Termination: The free end reaches the target height, which is constructed as:
#     `-cos(theta1) - cos(theta2 + theta1) > 1.0`
#     2. Truncation: Episode length is greater than 500 (200 for v0)
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box([ -1. -1. -1. -1. -12.566371 -28.274334], [ 1. 1. 1. 1. 12.566371 28.274334], (6,), float32)
# Action space:      Discrete(3)
# Max episode steps: 500
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Acrobot_v1.py --episodes 20            # evaluate
#   python Acrobot_v1.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Acrobot-v1"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return an integer action in [0, 3)
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
