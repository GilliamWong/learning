# ==========================================================================
# CarRacing-v3   (family: box2d)
# ==========================================================================
#     ## Description
#     The easiest control task to learn from pixels - a top-down
#     racing environment. The generated track is random every episode.
#
#     Some indicators are shown at the bottom of the window along with the
#     state RGB buffer. From left to right: true speed, four ABS sensors,
#     steering wheel position, and gyroscope.
#     To play yourself (it's rather fast for humans), type:
#     ```shell
#     python gymnasium/envs/box2d/car_racing.py
#     ```
#     Remember: it's a powerful rear-wheel drive car - don't press the accelerator
#     and turn at the same time.
#
#     ## Action Space
#     If continuous there are 3 actions :
#     - 0: steering, -1 is full left, +1 is full right
#     - 1: gas
#     - 2: braking
#
#     If discrete there are 5 actions:
#     - 0: do nothing
#     - 1: steer right
#     - 2: steer left
#     - 3: gas
#     - 4: brake
#
#     ## Observation Space
#
#     A top-down 96x96 RGB image of the car and race track.
#
#     ## Rewards
#     The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
#      visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
#
#     ## Starting State
#     The car starts at rest in the center of the road.
#
#     ## Episode Termination
#     The episode finishes when all the tiles are visited. The car can also go outside the playfield -
#      that is, far off the track, in which case it will receive -100 reward and die.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(0, 255, (96, 96, 3), uint8)
# Action space:      Box([-1. 0. 0.], 1.0, (3,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python CarRacing_v3.py --episodes 20            # evaluate
#   python CarRacing_v3.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "CarRacing-v3"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (3,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
