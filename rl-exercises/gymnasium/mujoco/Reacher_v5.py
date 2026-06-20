# ==========================================================================
# Reacher-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     "Reacher" is a two-jointed robot arm.
#     The goal is to move the robot's end effector (called *fingertip*) close to a target that is spawned at a random position.
#
#
#     ## Action Space
#     ```{figure} action_space_figures/reacher.png
#     :name: reacher
#     ```
#
#     The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.
#
#     | Num | Action                                                                          | Control Min | Control Max |Name (in corresponding XML file)| Joint | Type (Unit)  |
#     |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
#     | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1          | 1           | joint0                         | hinge | torque (N m) |
#     | 1   | Torque applied at the second hinge (connecting the two links)                   | -1          | 1           | joint1                         | hinge | torque (N m) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#
#     - *cos(qpos) (2 elements):* The cosine of the angles of the two arms.
#     - *sin(qpos) (2 elements):* The sine of the angles of the two arms.
#     - *qpos (2 elements):* The coordinates of the target.
#     - *qvel (2 elements):* The angular velocities of the arms (their derivatives).
#     - *xpos (2 elements):* The vector between the target and the reacher's.
#
#     The observation space is a `Box(-Inf, Inf, (10,), float64)` where the elements are as follows:
#
#     | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
#     | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
#     | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                      | hinge | unitless                 |
#     | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                      | hinge | unitless                 |
#     | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | sin(joint0)                      | hinge | unitless                 |
#     | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | sin(joint1)                      | hinge | unitless                 |
#     | 4   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                         | slide | position (m)             |
#     | 5   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                         | slide | position (m)             |
#     | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                           | hinge | angular velocity (rad/s) |
#     | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                           | hinge | angular velocity (rad/s) |
#     | 8   | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
#     | 9   | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
#     | excluded | z-value of position_fingertip - position_target (constantly 0 since reacher is 2d)        | -Inf | Inf | NA                               | slide | position (m)             |
#
#
#     Most Gymnasium environments just return the positions and velocities of the joints in the `.xml` file as the state of the environment.
#     In reacher, however, the state is created by combining only certain elements of the position and velocity and performing some function transformations on them.
#     The `reacher.xml` contains these 4 joints:
#
#     | Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
#     |-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
#     | 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
#     | 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
#     | 2   | x-coordinate of the target  | -Inf     | Inf      | target_x                         | slide | position (m)       |
#     | 3   | y-coordinate of the target  | -Inf     | Inf      | target_y                         | slide | position (m)       |
#
#
#     ## Rewards
#     The total reward is: ***reward*** *=* *reward_distance + reward_control*.
#
#     - *reward_distance*:
#     This reward is a measure of how far the *fingertip* of the reacher (the unattached end) is from the target,
#     with a more negative value assigned if the reacher's *fingertip* is further away from the target.
#     It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
#     where $w_{near}$ is the `reward_near_weight` (default is $1$).
#     - *reward_control*:
#     A negative reward to penalize the walker for taking actions that are too large.
#     It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
#     where $w_{control}$ is the `reward_control_weight`. (default is $0.1$)
#
#     `info` contains the individual reward terms.
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (10,), float64)
# Action space:      Box(-1.0, 1.0, (2,), float32)
# Max episode steps: 50
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Reacher_v5.py --episodes 20            # evaluate
#   python Reacher_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Reacher-v5"


class Policy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # TODO: set up parameters / Q-table / network / weights here

    def reset(self):
        # TODO (optional): reset any per-episode state
        pass

    def act(self, observation):
        # TODO: replace this random action with your own policy
        # return a numpy array, shape (2,), within the action bounds
        return self.action_space.sample()

    def update(self, obs, action, reward, next_obs, terminated, truncated, info):
        # TODO (optional): learn from this transition
        pass


if __name__ == "__main__":
    args = parse_args()
    run(ENV_ID, Policy, episodes=args.episodes, render=args.render, seed=args.seed)
