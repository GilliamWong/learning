# ==========================================================================
# Blackjack-v1   (family: toy_text)
# ==========================================================================
#     Blackjack is a card game where the goal is to beat the dealer by obtaining cards
#     that sum to closer to 21 (without going over 21) than the dealers cards.
#
#     ## Description
#     The game starts with the dealer having one face up and one face down card,
#     while the player has two face up cards. All cards are drawn from an infinite deck
#     (i.e. with replacement).
#
#     The card values are:
#     - Face cards (Jack, Queen, King) have a point value of 10.
#     - Aces can either count as 11 (called a 'usable ace') or 1.
#     - Numerical cards (2-10) have a value equal to their number.
#
#     The player has the sum of cards held. The player can request
#     additional cards (hit) until they decide to stop (stick) or exceed 21 (bust,
#     immediate loss).
#
#     After the player sticks, the dealer reveals their facedown card, and draws cards
#     until their sum is 17 or greater. If the dealer goes bust, the player wins.
#
#     If neither the player nor the dealer busts, the outcome (win, lose, draw) is
#     decided by whose sum is closer to 21.
#
#     This environment corresponds to the version of the blackjack problem
#     described in Example 5.1 in Reinforcement Learning: An Introduction
#     by Sutton and Barto [<a href="#blackjack_ref">1</a>].
#
#     ## Action Space
#     The action shape is `(1,)` in the range `{0, 1}` indicating
#     whether to stick or hit.
#
#     - 0: Stick
#     - 1: Hit
#
#     ## Observation Space
#     The observation consists of a 3-tuple containing: the player's current sum,
#     the value of the dealer's one showing card (1-10 where 1 is ace),
#     and whether the player holds a usable ace (0 or 1).
#
#     The observation is returned as `(int(), int(), int())`.
#
#     ## Starting State
#     The starting state is initialised with the following values.
#
#     | Observation               | Values         |
#     |---------------------------|----------------|
#     | Player current sum        |  4, 5, ..., 21 |
#     | Dealer showing card value |  1, 2, ..., 10 |
#     | Usable Ace                |  0, 1          |
#
#     ## Rewards
#     - win game: +1
#     - lose game: -1
#     - draw game: 0
#     - win game with natural blackjack:
#     +1.5 (if <a href="#nat">natural</a> is True)
#     +1 (if <a href="#nat">natural</a> is False)
#
#     ## Episode End
#     The episode ends if the following happens:
#
#     - Termination:
#     1. The player hits and the sum of hand exceeds 21.
#     2. The player sticks.
#
#     An ace will always be counted as usable (11) unless it busts the player.
#
#     ## Information
#
#     No additional information is returned.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Tuple(Discrete(32), Discrete(11), Discrete(2))
# Action space:      Discrete(2)
# Max episode steps: None
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Blackjack_v1.py --episodes 20            # evaluate
#   python Blackjack_v1.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Blackjack-v1"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return an integer action in [0, 2)
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
