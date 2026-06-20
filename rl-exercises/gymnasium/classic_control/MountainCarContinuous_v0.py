# ==========================================================================
# MountainCarContinuous-v0   (family: classic_control)
# ==========================================================================
#     ## Description
#
#     The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
#     at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
#     that can be applied to the car in either direction. The goal of the MDP is to strategically
#     accelerate the car to reach the goal state on top of the right hill. There are two versions
#     of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
#     This version is the one with continuous actions.
#
#     This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
#
#     ```
#     @TECHREPORT{Moore90efficientmemory-based,
#         author = {Andrew William Moore},
#         title = {Efficient Memory-based Learning for Robot Control},
#         institution = {University of Cambridge},
#         year = {1990}
#     }
#     ```
#
#     ## Observation Space
#
#     The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
#
#     | Num | Observation                          | Min   | Max  | Unit          |
#     |-----|--------------------------------------|-------|------|---------------|
#     | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m)  |
#     | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v)  |
#
#     ## Action Space
#
#     The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
#     The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.
#
#     ## Transition Dynamics:
#
#     Given an action, the mountain car follows the following transition dynamics:
#
#     *velocity<sub>t+1</sub> = velocity<sub>t</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*
#
#     *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
#
#     where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
#     The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
#     The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].
#
#     ## Reward
#
#     A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
#     taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
#     is added to the negative reward for that timestep.
#
#     ## Starting State
#
#     The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
#     The starting velocity of the car is always assigned to 0.
#
#     ## Episode End
#
#     The episode ends if either of the following happens:
#     1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
#     2. Truncation: The length of the episode is 999.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box([-1.2 -0.07], [0.6 0.07], (2,), float32)
# Action space:      Box(-1.0, 1.0, (1,), float32)
# Max episode steps: 999
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python MountainCarContinuous_v0.py --episodes 20            # evaluate
#   python MountainCarContinuous_v0.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "MountainCarContinuous-v0"


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
