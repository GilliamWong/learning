# ==========================================================================
# Pusher-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     "Pusher" is a multi-jointed robot arm that is very similar to a human arm.
#     The goal is to move a target cylinder (called *object*) to a goal position using the robot's end effector (called *fingertip*).
#     The robot consists of shoulder, elbow, forearm and wrist joints.
#
#
#     ## Action Space
#     ```{figure} action_space_figures/pusher.png
#     :name: pusher
#     ```
#
#     The action space is a `Box(-2, 2, (7,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.
#
#     | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
#     |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
#     | 0   | Rotation of the panning the shoulder                               | -2          | 2           | r_shoulder_pan_joint             | hinge | torque (N m) |
#     | 1   | Rotation of the shoulder lifting joint                             | -2          | 2           | r_shoulder_lift_joint            | hinge | torque (N m) |
#     | 2   | Rotation of the shoulder rolling joint                             | -2          | 2           | r_upper_arm_roll_joint           | hinge | torque (N m) |
#     | 3   | Rotation of hinge joint that flexed the elbow                      | -2          | 2           | r_elbow_flex_joint               | hinge | torque (N m) |
#     | 4   | Rotation of hinge that rolls the forearm                           | -2          | 2           | r_forearm_roll_joint             | hinge | torque (N m) |
#     | 5   | Rotation of flexing the wrist                                      | -2          | 2           | r_wrist_flex_joint               | hinge | torque (N m) |
#     | 6   | Rotation of rolling the wrist                                      | -2          | 2           | r_wrist_roll_joint               | hinge | torque (N m) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#
#     - *qpos (7 elements):* Position values of the robot's body parts.
#     - *qvel (7 elements):* The velocities of these individual body parts (their derivatives).
#     - *xpos (3 elements):* The coordinates of the fingertip of the pusher.
#     - *xpos (3 elements):* The coordinates of the object to be moved.
#     - *xpos (3 elements):* The coordinates of the goal position.
#
#     The observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
#
#     | Num | Observation                                              | Min  | Max | Name (in corresponding XML file) | Joint    | Type (Unit)              |
#     | --- | -------------------------------------------------------- | ---- | --- | -------------------------------- | -------- | ------------------------ |
#     | 0   | Rotation of the panning the shoulder                     | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angle (rad)              |
#     | 1   | Rotation of the shoulder lifting joint                   | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angle (rad)              |
#     | 2   | Rotation of the shoulder rolling joint                   | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angle (rad)              |
#     | 3   | Rotation of hinge joint that flexed the elbow            | -Inf | Inf | r_elbow_flex_joint               | hinge    | angle (rad)              |
#     | 4   | Rotation of hinge that rolls the forearm                 | -Inf | Inf | r_forearm_roll_joint             | hinge    | angle (rad)              |
#     | 5   | Rotation of flexing the wrist                            | -Inf | Inf | r_wrist_flex_joint               | hinge    | angle (rad)              |
#     | 6   | Rotation of rolling the wrist                            | -Inf | Inf | r_wrist_roll_joint               | hinge    | angle (rad)              |
#     | 7   | Rotational velocity of the panning the shoulder          | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angular velocity (rad/s) |
#     | 8   | Rotational velocity of the shoulder lifting joint        | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angular velocity (rad/s) |
#     | 9   | Rotational velocity of the shoulder rolling joint        | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angular velocity (rad/s) |
#     | 10  | Rotational velocity of hinge joint that flexed the elbow | -Inf | Inf | r_elbow_flex_joint               | hinge    | angular velocity (rad/s) |
#     | 11  | Rotational velocity of hinge that rolls the forearm      | -Inf | Inf | r_forearm_roll_joint             | hinge    | angular velocity (rad/s) |
#     | 12  | Rotational velocity of flexing the wrist                 | -Inf | Inf | r_wrist_flex_joint               | hinge    | angular velocity (rad/s) |
#     | 13  | Rotational velocity of rolling the wrist                 | -Inf | Inf | r_wrist_roll_joint               | hinge    | angular velocity (rad/s) |
#     | 14  | x-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
#     | 15  | y-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
#     | 16  | z-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
#     | 17  | x-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidex)              | slide    | position (m)             |
#     | 18  | y-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidey)              | slide    | position (m)             |
#     | 19  | z-coordinate of the object to be moved                   | -Inf | Inf | object                           | cylinder | position (m)             |
#     | 20  | x-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidex)               | slide    | position (m)             |
#     | 21  | y-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidey)               | slide    | position (m)             |
#     | 22  | z-coordinate of the goal position of the object          | -Inf | Inf | goal                             | sphere   | position (m)             |
#
#     To understand the state space, an analogy can be drawn to a human arm, where the words "flex" and "roll" have the same meaning as in human joints.
#
#     ## Rewards
#     The total reward is: ***reward*** *=* *reward_dist + reward_ctrl + reward_near*.
#
#     - *reward_dist*:
#     This reward is a measure of how far the object is from the target goal position,
#     with a more negative value assigned if the object is further away from the target.
#     It is $-w_{dist} \|(P_{object} - P_{target})\|_2$.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (23,), float64)
# Action space:      Box(-2.0, 2.0, (7,), float32)
# Max episode steps: 100
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Pusher_v5.py --episodes 20            # evaluate
#   python Pusher_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Pusher-v5"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (7,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
