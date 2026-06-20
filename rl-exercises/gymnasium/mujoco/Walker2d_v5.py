# ==========================================================================
# Walker2d-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment builds on the [hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) environment by adding another set of legs that allow the robot to walk forward instead of hop.
#     Like other MuJoCo environments, this environment aims to increase the number of independent state and control variables compared to classical control environments.
#     The walker is a two-dimensional bipedal robot consisting of seven main body parts - a single torso at the top (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs below the thighs, and two feet attached to the legs on which the entire body rests.
#     The goal is to walk in the forward (right) direction by applying torque to the six hinges connecting the seven body parts.
#
#
#     ## Action Space
#     ```{figure} action_space_figures/walker2d.png
#     :name: walker2d
#     ```
#
#     The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
#
#     | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
#     |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
#     | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
#     | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
#     | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
#     | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
#     | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
#     | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#
#     - *qpos (8 elements by default):* Position values of the robot's body parts.
#     - *qvel (9 elements):* The velocities of these individual body parts (their derivatives).
#
#     By default, the observation does not include the robot's x-coordinate (`rootx`).
#     This can be included by passing `exclude_current_positions_from_observation=False` during construction.
#     In this case, the observation space will be a `Box(-Inf, Inf, (18,), float64)`, where the first observation element is the x-coordinate of the robot.
#     Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x-coordinate are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
#
#     By default, however, the observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
#
#     | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
#     | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
#     | 0   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
#     | 1   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
#     | 2   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
#     | 3   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
#     | 4   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
#     | 5   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
#     | 6   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
#     | 7   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
#     | 8   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
#     | 9   | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
#     | 10  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
#     | 11  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
#     | 12  | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
#     | 13  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
#     | 14  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
#     | 15  | angular velocity of the leg hinge                  | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
#     | 16  | angular velocity of the foot hinge                 | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |
#     | excluded | x-coordinate of the torso                     | -Inf | Inf | rootx                            | slide | position (m)             |
#
#
#     ## Rewards
#     The total reward is: ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost*.
#
#     - *healthy_reward*:
#     Every timestep that the Walker2d is alive, it receives a fixed reward of value `healthy_reward` (default is $1$),
#     - *forward_reward*:
#     A reward for moving forward,
#     this reward would be positive if the Walker2d moves forward (in the positive $x$ direction / in the right direction).
#     $w_{forward} \times \frac{dx}{dt}$, where
#     $dx$ is the displacement of the (front) "tip" ($x_{after-action} - x_{before-action}$),
#     $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $4$),
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (17,), float64)
# Action space:      Box(-1.0, 1.0, (6,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Walker2d_v5.py --episodes 20            # evaluate
#   python Walker2d_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Walker2d-v5"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (6,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
