# ==========================================================================
# InvertedDoublePendulum-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment originates from control theory and builds on the cartpole environment based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
#     powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity or constraints).
#     This environment involves a cart that can be moved linearly, with one pole attached to it and a second pole attached to the other end of the first pole (leaving the second pole as the only one with a free end).
#     The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces to the cart.
#
#
#     ## Action Space
#     The agent take a 1-element vector for actions.
#     The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
#     numerical force applied to the cart (with magnitude representing the amount of force and
#     sign representing the direction)
#
#     | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
#     |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
#     | 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#
#     - *qpos (1 element):* Position values of the robot's cart.
#     - *sin(qpos) (2 elements):* The sine of the angles of poles.
#     - *cos(qpos) (2 elements):* The cosine of the angles of poles.
#     - *qvel (3 elements):* The velocities of these individual body parts (their derivatives).
#     - *qfrc_constraint (1 element):* Constraint force of the cart.
#     There is one constraint force for contacts for each degree of freedom (3).
#     The approach and handling of constraints by MuJoCo is unique to the simulator and is based on their research.
#     More information can be found  in their [*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html) or in their paper ["Analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo"](https://homes.cs.washington.edu/~todorov/papers/TodorovICRA14.pdf).
#
#     The observation space is a `Box(-Inf, Inf, (9,), float64)` where the elements are as follows:
#
#     | Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
#     | --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
#     | 0   | position of the cart along the linear surface                     | -Inf | Inf | slider                           | slide | position (m)             |
#     | 1   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                       | hinge | unitless                 |
#     | 2   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                      | hinge | unitless                 |
#     | 3   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                       | hinge | unitless                 |
#     | 4   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                      | hinge | unitless                 |
#     | 5   | velocity of the cart                                              | -Inf | Inf | slider                           | slide | velocity (m/s)           |
#     | 6   | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s) |
#     | 7   | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                           | hinge | angular velocity (rad/s) |
#     | 8   | constraint force - x                                              | -Inf | Inf | slider                           | slide | Force (N)                |
#     | excluded | constraint force - y                                         | -Inf | Inf | slider                           | slide | Force (N)                |
#     | excluded | constraint force - z                                         | -Inf | Inf | slider                           | slide | Force (N)                |
#
#
#     ## Rewards
#     The total reward is: ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*.
#
#     - *alive_bonus*:
#     Every timestep that the Inverted Pendulum is healthy (see definition in section "Episode End"),
#     it gets a reward of fixed value `healthy_reward` (default is $10$).
#     - *distance_penalty*:
#     This reward is a measure of how far the *tip* of the second pendulum (the only free end) moves,
#     and it is calculated as $0.01 x_{pole2-tip}^2 + (y_{pole2-tip}-2)^2$,
#     where $x_{pole2-tip}, y_{pole2-tip}$ are the xy-coordinatesof the tip of the second pole.
#     - *velocity_penalty*:
#     A negative reward to penalize the agent for moving too fast.
#     $10^{-3} \omega_1 + 5 \times 10^{-3} \omega_2$,
#     where $\omega_1, \omega_2$ are the angular velocities of the hinges.
#
#     `info` contains the individual reward terms.
#
#
#     ## Starting State
#     The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{3}, reset\_noise\_scale \times I_{3}]}$.
#     The initial velocity state is $\mathcal{N}(0_{3}, reset\_noise\_scale^2 \times I_{3})$.
#
#     where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (9,), float64)
# Action space:      Box(-1.0, 1.0, (1,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python InvertedDoublePendulum_v5.py --episodes 20            # evaluate
#   python InvertedDoublePendulum_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "InvertedDoublePendulum-v5"


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
