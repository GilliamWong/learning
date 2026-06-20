# ==========================================================================
# Taxi-v4   (family: toy_text)
# ==========================================================================
#     The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them
#     off at one of four locations.
#
#     ## Description
#     There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the
#     5x5 grid world. The taxi starts off at a random square and the passenger at one of the
#     designated locations.
#
#     The goal is move the taxi to the passenger's location, pick up the passenger,
#     move to the passenger's desired destination, and
#     drop off the passenger. Once the passenger is dropped off, the episode ends.
#
#     The player receives positive rewards for successfully dropping-off the passenger at the correct
#     location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and
#     for each step where another reward is not received.
#
#     Map:
#
#         +---------+
#         |R: | : :G|
#         | : | : : |
#         | : : : : |
#         | | : | : |
#         |Y| : |B: |
#         +---------+
#
#     From "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
#     by Tom Dietterich [<a href="#taxi_ref">1</a>].
#
#     ## Action Space
#     The action shape is `(1,)` in the range `{0, 5}` indicating
#     which direction to move the taxi or to pickup/drop off passengers.
#
#     - 0: Move south (down)
#     - 1: Move north (up)
#     - 2: Move east (right)
#     - 3: Move west (left)
#     - 4: Pickup passenger
#     - 5: Drop off passenger
#
#     ## Observation Space
#     There are 500 discrete states since there are 25 taxi positions, 5 possible
#     locations of the passenger (including the case when the passenger is in the
#     taxi), and 4 destination locations.
#
#     Destination on the ansi rendered map are represented with the first letter of the color.
#
#     Passenger locations:
#     - 0: Red
#     - 1: Green
#     - 2: Yellow
#     - 3: Blue
#     - 4: In taxi
#
#     Destinations:
#     - 0: Red
#     - 1: Green
#     - 2: Yellow
#     - 3: Blue
#
#     An observation is returned as an `int()` that encodes the corresponding state, calculated by
#     `((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination`
#
#     Note that there are 400 states that can actually be reached during an
#     episode. The missing states correspond to situations in which the passenger
#     is at the same location as their destination, as this typically signals the
#     end of an episode. Four additional states can be observed right after a
#     successful episodes, when both the passenger and the taxi are at the destination.
#     This gives a total of 404 reachable discrete states.
#
#
# ---- quick reference ----------------------------------------------------
# Observation space: Discrete(500)
# Action space:      Discrete(6)
# Max episode steps: 200
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Taxi_v4.py --episodes 20            # evaluate
#   python Taxi_v4.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Taxi-v4"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return an integer action in [0, 6)
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
