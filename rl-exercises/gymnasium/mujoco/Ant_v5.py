# ==========================================================================
# Ant-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment is based on the one introduced by Schulman, Moritz, Levine, Jordan, and Abbeel in ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).
#     The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts.
#     The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).
#
#     Note: Although the robot is called "Ant", it is actually 75cm tall and weighs 910.88g, with the torso being 327.25g and each leg being 145.91g.
#
#     ## Action Space
#     ```{figure} action_space_figures/ant.png
#     :name: ant
#     ```
#
#     The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.
#
#     | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
#     | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
#     | 0   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
#     | 1   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |
#     | 2   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
#     | 3   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
#     | 4   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
#     | 5   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |
#     | 6   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
#     | 7   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#
#     - *qpos (13 elements by default):* Position values of the robot's body parts.
#     - *qvel (14 elements):* The velocities of these individual body parts (their derivatives).
#     - *cfrc_ext (78 elements):* This is the center of mass based external forces on the body parts.
#     It has shape 13 * 6 (*nbody * 6*) and hence adds another 78 elements to the state space.
#     (external forces - force x, y, z and torque x, y, z)
#
#     By default, the observation does not include the x- and y-coordinates of the torso.
#     These can be included by passing `exclude_current_positions_from_observation=False` during construction.
#     In this case, the observation space will be a `Box(-Inf, Inf, (107,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
#     Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
#
#     By default, however, the observation space is a `Box(-Inf, Inf, (105,), float64)`, where the position and velocity elements are as follows:
#
#     | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Type (Unit)              |
#     |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
#     | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | root                                   | free  | position (m)             |
#     | 1   | w-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
#     | 2   | x-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
#     | 3   | y-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
#     | 4   | z-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
#     | 5   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
#     | 6   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
#     | 7   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
#     | 8   | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
#     | 9   | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
#     | 10  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
#     | 11  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
#     | 12  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
#     | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
#     | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
#     | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
#     | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
#     | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
#     | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
#     | 19  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
#     | 20  | angular velocity of the angle between front left links       | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
#     | 21  | angular velocity of angle between torso and front right link | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
#     | 22  | angular velocity of the angle between front right links      | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
#     | 23  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
#     | 24  | angular velocity of the angle between back left links        | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
#     | 25  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (105,), float64)
# Action space:      Box(-1.0, 1.0, (8,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Ant_v5.py --episodes 20            # evaluate
#   python Ant_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Ant-v5"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (8,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
