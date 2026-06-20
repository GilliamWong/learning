# ==========================================================================
# InvertedPendulum-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment is the Cartpole environment, based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
#     just like in the classic environments, but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity).
#     This environment consists of a cart that can be moved linearly, with a pole attached to one end and having another end free.
#     The cart can be pushed left or right, and the goal is to balance the pole on top of the cart by applying forces to the cart.
#
#
#     ## Action Space
#     The agent take a 1-element vector for actions.
#
#     The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
#     the numerical force applied to the cart (with magnitude representing the amount of
#     force and sign representing the direction)
#
#     | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
#     |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
#     | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#     - *qpos (2 element):* Position values of the robot's cart and pole.
#     - *qvel (2 elements):* The velocities of cart and pole (their derivatives).
#
#     The observation space is a `Box(-Inf, Inf, (4,), float64)` where the elements are as follows:
#
#     | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
#     | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
#     | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
#     | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
#     | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
#     | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s)  |
#
#
#     ## Rewards
#     The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.
#
#     The pole is considered upright if:
#     $|angle| < 0.2$.
#
#     and `info` also contains the reward.
#
#
#     ## Starting State
#     The initial position state is $\mathcal{U}_{[-reset\_noise\_scale 	imes I_{2}, reset\_noise\_scale 	imes I_{2}]}$.
#     The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale 	imes I_{2}, reset\_noise\_scale 	imes I_{2}]}$.
#
#     where $\mathcal{U}$ is the multivariate uniform continuous distribution.
#
#
#     ## Episode End
#     ### Termination
#     The environment terminates when the Inverted Pendulum is unhealthy.
#     The Inverted Pendulum is unhealthy if any of the following happens:
#
#     1. Any of the state space values is no longer finite.
#     2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radians.
#
#     ### Truncation
#     The default duration of an episode is 1000 timesteps.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (4,), float64)
# Action space:      Box(-3.0, 3.0, (1,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python InvertedPendulum_v5.py --episodes 20            # evaluate
#   python InvertedPendulum_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "InvertedPendulum-v5"


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
