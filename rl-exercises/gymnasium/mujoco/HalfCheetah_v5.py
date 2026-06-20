# ==========================================================================
# HalfCheetah-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment is based on the work of P. Wawrzyński in ["A Cat-Like Robot Real-Time Learning to Run"](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).
#     The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws).
#     The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
#     The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).
#
#
#     ## Action Space
#     ```{figure} action_space_figures/half_cheetah.png
#     :name: half_cheetah
#     ```
#
#     The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
#
#     | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
#     | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
#     | 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
#     | 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
#     | 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
#     | 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
#     | 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
#     | 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |
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
#     Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
#
#     By default, however, the observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
#
#
#     | Num | Observation                                 | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
#     | --- | ------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
#     | 0   | z-coordinate of the front tip               | -Inf | Inf | rootz                            | slide | position (m)             |
#     | 1   | angle of the front tip                      | -Inf | Inf | rooty                            | hinge | angle (rad)              |
#     | 2   | angle of the back thigh                     | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
#     | 3   | angle of the back shin                      | -Inf | Inf | bshin                            | hinge | angle (rad)              |
#     | 4   | angle of the back foot                      | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
#     | 5   | angle of the front thigh                    | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
#     | 6   | angle of the front shin                     | -Inf | Inf | fshin                            | hinge | angle (rad)              |
#     | 7   | angle of the front foot                     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
#     | 8   | velocity of the x-coordinate of front tip   | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
#     | 9   | velocity of the z-coordinate of front tip   | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
#     | 10  | angular velocity of the front tip           | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
#     | 11  | angular velocity of the back thigh          | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
#     | 12  | angular velocity of the back shin           | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
#     | 13  | angular velocity of the back foot           | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
#     | 14  | angular velocity of the front thigh         | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
#     | 15  | angular velocity of the front shin          | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
#     | 16  | angular velocity of the front foot          | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |
#     | excluded | x-coordinate of the front tip          | -Inf | Inf | rootx                            | slide | position (m)             |
#
#
#     ## Rewards
#     The total reward is: ***reward*** *=* *forward_reward - ctrl_cost*.
#
#     - *forward_reward*:
#     A reward for moving forward,
#     this reward would be positive if the Half Cheetah moves forward (in the positive $x$ direction / in the right direction).
#     $w_{forward} \times \frac{dx}{dt}$, where
#     $dx$ is the displacement of the "tip" ($x_{after-action} - x_{before-action}$),
#     $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
#     and `frametime` which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (17,), float64)
# Action space:      Box(-1.0, 1.0, (6,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python HalfCheetah_v5.py --episodes 20            # evaluate
#   python HalfCheetah_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "HalfCheetah-v5"


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
