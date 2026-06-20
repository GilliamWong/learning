# ==========================================================================
# MountainCar-v0   (family: classic_control)
# ==========================================================================
#     ## Description
#
#     The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
#     at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
#     that can be applied to the car in either direction. The goal of the MDP is to strategically
#     accelerate the car to reach the goal state on top of the right hill. There are two versions
#     of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
#     This version is the one with discrete actions.
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
#     | Num | Observation                          | Min   | Max  | Unit         |
#     |-----|--------------------------------------|-------|------|--------------|
#     | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m) |
#     | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v) |
#
#     ## Action Space
#
#     There are 3 discrete deterministic actions:
#
#     - 0: Accelerate to the left
#     - 1: Don't accelerate
#     - 2: Accelerate to the right
#
#     ## Transition Dynamics:
#
#     Given an action, the mountain car follows the following transition dynamics:
#
#     *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*
#
#     *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
#
#     where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
#     upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
#     velocity is clipped to the range `[-0.07, 0.07]`.
#
#     ## Reward:
#
#     The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
#     penalised with a reward of -1 for each timestep.
#
#     ## Starting State
#
#     The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
#     The starting velocity of the car is always assigned to 0.
#
#     ## Episode End
#
#     The episode ends if either of the following happens:
#     1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
#     2. Truncation: The length of the episode is 200.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box([-1.2 -0.07], [0.6 0.07], (2,), float32)
# Action space:      Discrete(3)
# Max episode steps: 200
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python MountainCar_v0.py --episodes 20            # evaluate
#   python MountainCar_v0.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "MountainCar-v0"


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
