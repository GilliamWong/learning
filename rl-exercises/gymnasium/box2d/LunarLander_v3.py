# ==========================================================================
# LunarLander-v3   (family: box2d)
# ==========================================================================
#     ## Description
#     This environment is a classic rocket trajectory optimization problem.
#     According to Pontryagin's maximum principle, it is optimal to fire the
#     engine at full throttle or turn it off. This is the reason why this
#     environment has discrete actions: engine on or off.
#
#     There are two environment versions: discrete or continuous.
#     The landing pad is always at coordinates (0,0). The coordinates are the
#     first two numbers in the state vector.
#     Landing outside of the landing pad is possible. Fuel is infinite, so an agent
#     can learn to fly and then land on its first attempt.
#
#     To see a heuristic landing, run:
#     ```shell
#     python gymnasium/envs/box2d/lunar_lander.py
#     ```
#
#     ## Action Space
#     There are four discrete actions available:
#     - 0: do nothing
#     - 1: fire left orientation engine
#     - 2: fire main engine
#     - 3: fire right orientation engine
#
#     ## Observation Space
#     The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
#     velocities in `x` & `y`, its angle, its angular velocity, and two booleans
#     that represent whether each leg is in contact with the ground or not.
#
#     ## Rewards
#     After every step a reward is granted. The total reward of an episode is the
#     sum of the rewards for all the steps within that episode.
#
#     For each step, the reward:
#     - is increased/decreased the closer/further the lander is to the landing pad.
#     - is increased/decreased the slower/faster the lander is moving.
#     - is decreased the more the lander is tilted (angle not horizontal).
#     - is increased by 10 points for each leg that is in contact with the ground.
#     - is decreased by 0.03 points each frame a side engine is firing.
#     - is decreased by 0.3 points each frame the main engine is firing.
#
#     The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.
#
#     An episode is considered a solution if it scores at least 200 points.
#
#     ## Starting State
#     The lander starts at the top center of the viewport with a random initial
#     force applied to its center of mass.
#
#     ## Episode Termination
#     The episode finishes if:
#     1) the lander crashes (the lander body gets in contact with the moon);
#     2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
#     3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
#         a body which is not awake is a body which doesn't move and doesn't
#         collide with any other body:
#     > When Box2D determines that a body (or group of bodies) has come to rest,
#     > the body enters a sleep state which has very little CPU overhead. If a
#     > body is awake and collides with a sleeping body, then the sleeping body
#     > wakes up. Bodies will also wake up if a joint or contact attached to
#     > them is destroyed.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
# Action space:      Discrete(4)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python LunarLander_v3.py --episodes 20            # evaluate
#   python LunarLander_v3.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "LunarLander-v3"


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
