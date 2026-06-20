# ==========================================================================
# BipedalWalkerHardcore-v3   (family: box2d)
# ==========================================================================
#     ## Description
#     This is a simple 4-joint walker robot environment.
#     There are two versions:
#     - Normal, with slightly uneven terrain.
#     - Hardcore, with ladders, stumps, pitfalls.
#
#     To solve the normal version, you need to get 300 points in 1600 time steps.
#     To solve the hardcore version, you need 300 points in 2000 time steps.
#
#     A heuristic is provided for testing. It's also useful to get demonstrations
#     to learn from. To run the heuristic:
#     ```
#     python gymnasium/envs/box2d/bipedal_walker.py
#     ```
#
#     ## Action Space
#     Actions are motor speed values in the [-1, 1] range for each of the
#     4 joints at both hips and knees.
#
#     ## Observation Space
#     State consists of hull angle speed, angular velocity, horizontal speed,
#     vertical speed, position of joints and joints angular speed, legs contact
#     with ground, and 10 lidar rangefinder measurements. There are no coordinates
#     in the state vector.
#
#     ## Rewards
#     Reward is given for moving forward, totaling 300+ points up to the far end.
#     If the robot falls, it gets -100. Applying motor torque costs a small
#     amount of points. A more optimal agent will get a better score.
#
#     ## Starting State
#     The walker starts standing at the left end of the terrain with the hull
#     horizontal, and both legs in the same position with a slight knee angle.
#
#     ## Episode Termination
#     The episode will terminate if the hull gets in contact with the ground or
#     if the walker exceeds the right end of the terrain length.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box([-3.1415927 -5. -5. -5. -3.1415927 -5. -3.1415927 -5. -0. -3.1415927 -5. -3.1415927 -5. -0. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. ], [3.1415927 5. 5. 5. 3.1415927 5. 3.1415927 5. 5. 3.1415927 5. 3.1415927 5. 5. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ], (24,), float32)
# Action space:      Box(-1.0, 1.0, (4,), float32)
# Max episode steps: 2000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python BipedalWalkerHardcore_v3.py --episodes 20            # evaluate
#   python BipedalWalkerHardcore_v3.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "BipedalWalkerHardcore-v3"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (4,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
