# ==========================================================================
# CliffWalkingSlippery-v1   (family: toy_text)
# ==========================================================================
#     Cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff.
#
#     ## Description
#     The game starts with the player at location [3, 0] of the 4x12 grid world with the
#     goal located at [3, 11]. If the player reaches the goal the episode ends.
#
#     A cliff runs along [3, 1..10]. If the player moves to a cliff location it
#     returns to the start location.
#
#     The player makes moves until they reach the goal.
#
#     Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
#     by Sutton and Barto [<a href="#cliffwalk_ref">1</a>].
#
#     The cliff can be chosen to be slippery (disabled by default) so the player may move perpendicular
#     to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).
#
#     With inspiration from:
#     [https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)
#
#     ## Action Space
#     The action shape is `(1,)` in the range `{0, 3}` indicating
#     which direction to move the player.
#
#     - 0: Move up
#     - 1: Move right
#     - 2: Move down
#     - 3: Move left
#
#     ## Observation Space
#     There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at
#     the goal as the latter results in the end of the episode. What remains are all
#     the positions of the first 3 rows plus the bottom-left cell.
#
#     The observation is a value representing the player's current position as
#     current_row * ncols + current_col (where both the row and col start at 0).
#
#     For example, the starting position can be calculated as follows: 3 * 12 + 0 = 36.
#
#     The observation is returned as an `int()`.
#
#     ## Starting State
#     The episode starts with the player in state `[36]` (location [3, 0]).
#
#     ## Reward
#     Each time step incurs -1 reward, unless the player stepped into the cliff,
#     which incurs -100 reward.
#
#     ## Episode End
#     The episode terminates when the player enters state `[47]` (location [3, 11]).
#
#     ## Information
#
#     `step()` and `reset()` return a dict with the following keys:
#     - "p" - transition proability for the state.
#
#     As cliff walking is not stochastic, the transition probability returned always 1.0.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Discrete(48)
# Action space:      Discrete(4)
# Max episode steps: None
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python CliffWalkingSlippery_v1.py --episodes 20            # evaluate
#   python CliffWalkingSlippery_v1.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "CliffWalkingSlippery-v1"


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
