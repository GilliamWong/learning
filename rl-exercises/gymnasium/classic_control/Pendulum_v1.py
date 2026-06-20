# ==========================================================================
# Pendulum-v1   (family: classic_control)
# ==========================================================================
#     ## Description
#
#     The inverted pendulum swingup problem is based on the classic problem in control theory.
#     The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
#     The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
#     into an upright position, with its center of gravity right above the fixed point.
#
#     The diagram below specifies the coordinate system used for the implementation of the pendulum's
#     dynamic equations.
#
#     ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)
#
#     - `x-y`: cartesian coordinates of the pendulum's end in meters.
#     - `theta` : angle in radians.
#     - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
#
#     ## Action Space
#
#     The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.
#
#     | Num | Action | Min  | Max |
#     |-----|--------|------|-----|
#     | 0   | Torque | -2.0 | 2.0 |
#
#     ## Observation Space
#
#     The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
#     end and its angular velocity.
#
#     | Num | Observation      | Min  | Max |
#     |-----|------------------|------|-----|
#     | 0   | x = cos(theta)   | -1.0 | 1.0 |
#     | 1   | y = sin(theta)   | -1.0 | 1.0 |
#     | 2   | Angular Velocity | -8.0 | 8.0 |
#
#     ## Rewards
#
#     The reward function is defined as:
#
#     *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
#
#     where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
#     Based on the above equation, the minimum reward that can be obtained is
#     *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
#     while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
#
#     ## Starting State
#
#     The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.
#
#     ## Episode Truncation
#
#     The episode truncates at 200 time steps.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
# Action space:      Box(-2.0, 2.0, (1,), float32)
# Max episode steps: 200
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Pendulum_v1.py --episodes 20            # evaluate
#   python Pendulum_v1.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Pendulum-v1"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (1,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
