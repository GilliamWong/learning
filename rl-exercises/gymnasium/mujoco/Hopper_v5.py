# ==========================================================================
# Hopper-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment is based on the work of Erez, Tassa, and Todorov in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf).
#     The environment aims to increase the number of independent state and control variables compared to classical control environments.
#     The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
#     The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts.
#
#
#     ## Action Space
#     ```{figure} action_space_figures/hopper.png
#     :name: hopper
#     ```
#
#     The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied at the hinge joints.
#
#     | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
#     |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
#     | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
#     | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
#     | 2   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#
#     - *qpos (5 elements by default):* Position values of the robot's body parts.
#     - *qvel (6 elements):* The velocities of these individual body parts (their derivatives).
#
#     By default, the observation does not include the robot's x-coordinate (`rootx`).
#     This can  be included by passing `exclude_current_positions_from_observation=False` during construction.
#     In this case, the observation space will be a `Box(-Inf, Inf, (12,), float64)`, where the first observation element is the x-coordinate of the robot.
#     Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
#
#     By default, however, the observation space is a `Box(-Inf, Inf, (11,), float64)` where the elements are as follows:
#
#     | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
#     | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
#     | 0   | z-coordinate of the torso (height of hopper)       | -Inf | Inf | rootz                            | slide | position (m)             |
#     | 1   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
#     | 2   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
#     | 3   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
#     | 4   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
#     | 5   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
#     | 6   | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
#     | 7   | angular velocity of the angle of the torso         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
#     | 8   | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
#     | 9   | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
#     | 10  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
#     | excluded | x-coordinate of the torso                     | -Inf | Inf | rootx                            | slide | position (m)             |
#
#
#     ## Rewards
#     The total reward is: ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.
#
#     - *healthy_reward*:
#     Every timestep that the Hopper is healthy (see definition in section "Episode End"),
#     it gets a reward of fixed value `healthy_reward` (default is $1$).
#     - *forward_reward*:
#     A reward for moving forward,
#     this reward would be positive if the Hopper moves forward (in the positive $x$ direction / in the right direction).
#     $w_{forward} \times \frac{dx}{dt}$, where
#     $dx$ is the displacement of the "torso" ($x_{after-action} - x_{before-action}$),
#     $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $4$),
#     and `frametime` which is $0.002$ - so the default is $dt = 4 \times 0.002 = 0.008$,
#     $w_{forward}$ is the `forward_reward_weight` (default is $1$).
#     - *ctrl_cost*:
#     A negative reward to penalize the Hopper for taking actions that are too large.
#     $w_{control} \times \|action\|_2^2$,
#     where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).
#
#     `info` contains the individual reward terms.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (11,), float64)
# Action space:      Box(-1.0, 1.0, (3,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Hopper_v5.py --episodes 20            # evaluate
#   python Hopper_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Hopper-v5"


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
