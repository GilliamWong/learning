# ==========================================================================
# HumanoidStandup-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment is based on the environment introduced by Tassa, Erez and Todorov in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
#     The 3D bipedal robot is designed to simulate a human.
#     It has a torso (abdomen) with a pair of legs and arms, and a pair of tendons connecting the hips to the knees.
#     The legs each consist of three body parts (thigh, shin, foot), and the arms consist of two body parts (upper arm, forearm).
#     The environment starts with the humanoid laying on the ground, and then the goal of the environment is to make the humanoid stand up and then keep it standing by applying torques to the various hinges.
#
#
#     ## Action Space
#     ```{figure} action_space_figures/humanoid.png
#     :name: humanoid
#     ```
#
#     The action space is a `Box(-0.4, 0.4, (17,), float32)`. An action represents the torques applied at the hinge joints.
#
#     | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
#     | --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
#     | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_y                        | hinge | torque (N m) |
#     | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_z                        | hinge | torque (N m) |
#     | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_x                        | hinge | torque (N m) |
#     | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | right_hip_x (right_thigh)        | hinge | torque (N m) |
#     | 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | right_hip_z (right_thigh)        | hinge | torque (N m) |
#     | 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | right_hip_y (right_thigh)        | hinge | torque (N m) |
#     | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | right_knee                       | hinge | torque (N m) |
#     | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | left_hip_x (left_thigh)          | hinge | torque (N m) |
#     | 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | left_hip_z (left_thigh)          | hinge | torque (N m) |
#     | 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | left_hip_y (left_thigh)          | hinge | torque (N m) |
#     | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | left_knee                        | hinge | torque (N m) |
#     | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | right_shoulder1                  | hinge | torque (N m) |
#     | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | right_shoulder2                  | hinge | torque (N m) |
#     | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | right_elbow                      | hinge | torque (N m) |
#     | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | left_shoulder1                   | hinge | torque (N m) |
#     | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | left_shoulder2                   | hinge | torque (N m) |
#     | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | left_elbow                       | hinge | torque (N m) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order)
#
#     - *qpos (22 elements by default):* The position values of the robot's body parts.
#     - *qvel (23 elements):* The velocities of these individual body parts (their derivatives).
#     - *cinert (130 elements):* Mass and inertia of the rigid body parts relative to the center of mass,
#     (this is an intermediate result of the transition).
#     It has shape 13*10 (*nbody * 10*).
#     (cinert - inertia matrix and body mass offset and body mass)
#     - *cvel (78 elements):* Center of mass based velocity.
#     It has shape 13 * 6 (*nbody * 6*).
#     (com velocity - velocity x, y, z and angular velocity x, y, z)
#     - *qfrc_actuator (17 elements):* Constraint force generated as the actuator force at each joint.
#     This has shape `(17,)`  *(nv * 1)*.
#     - *cfrc_ext (78 elements):* This is the center of mass based external force on the body parts.
#     It has shape 13 * 6 (*nbody * 6*) and thus adds another 78 elements to the observation space.
#     (external forces - force x, y, z and torque x, y, z)
#
#     where *nbody* is the number of bodies in the robot,
#     and *nv* is the number of degrees of freedom (*= dim(qvel)*).
#
#     By default, the observation does not include the x- and y-coordinates of the torso.
#     These can be included by passing `exclude_current_positions_from_observation=False` during construction.
#     In this case, the observation space will be a `Box(-Inf, Inf, (350,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
#     Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
#
#     By default, however, the observation space is a `Box(-Inf, Inf, (348,), float64)`, where the position and velocity elements are as follows:
#
#     | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)                |
#     | --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
#     | 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
#     | 1   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
#     | 2   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
#     | 3   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (348,), float64)
# Action space:      Box(-0.4, 0.4, (17,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python HumanoidStandup_v5.py --episodes 20            # evaluate
#   python HumanoidStandup_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "HumanoidStandup-v5"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (17,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
