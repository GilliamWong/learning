# ==========================================================================
# FrozenLake-v1   (family: toy_text)
# ==========================================================================
#      Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
#      by walking over the frozen lake.
#      The player may not always move in the intended direction due to the slippery nature of the frozen lake.
#
#      ## Description
#      The game starts with the player at location `[0,0]` of the frozen lake grid world with the
#      goal located at far extent of the world e.g. `[3,3]` for the 4x4 environment.
#
#      Holes in the ice are distributed in set locations when using a pre-determined map
#      or in random locations when a random map is generated.
#      Randomly generated worlds will always have a path to the goal.
#
#      The player makes moves until they reach the goal or fall in a hole.
#
#      The lake is slippery (unless disabled) so the player may move perpendicular
#      to the intended direction sometimes (see `is_slippery` in Argument section).
#
#      Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
#      All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).
#
#      ## Action Space
#      The action shape is `(1,)` in the range `{0, 3}` indicating
#      which direction to move the player.
#
#      - 0: Move left
#      - 1: Move down
#      - 2: Move right
#      - 3: Move up
#
#      ## Observation Space
#      The observation is a value representing the player's current position as
#      `current_row * ncols + current_col` (where both the row and col start at 0).
#      Therefore, the observation is returned as an integer.
#
#      For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
#      The number of possible observations is dependent on the size of the map.
#
#      ## Starting State
#      The episode starts with the player in state `[0]` (location [0, 0]).
#
#      ## Rewards
#
#      Default reward schedule:
#      - Reach goal: +1
#      - Reach hole: 0
#      - Reach frozen: 0
#
#      See `reward_schedule` for reward customization in the Argument section.
#
#      ## Episode End
#      The episode ends if the following happens:
#
#      - Termination:
#          1. The player moves into a hole.
#          2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).
#
#      - Truncation (using the time_limit wrapper):
#          1. The length of the episode is 100 for FrozenLake4x4, 200 for FrozenLake8x8.
#
#      ## Information
#
#      `step()` and `reset()` return a dict with the following keys:
#      - `p`: transition probability for the state which will be impacted by the `is_slippery` parameter.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Discrete(16)
# Action space:      Discrete(4)
# Max episode steps: 100
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python FrozenLake_v1.py --episodes 20            # evaluate
#   python FrozenLake_v1.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "FrozenLake-v1"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return an integer action in [0, 4)
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
