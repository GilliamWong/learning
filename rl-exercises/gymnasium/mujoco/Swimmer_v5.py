# ==========================================================================
# Swimmer-v5   (family: mujoco)
# ==========================================================================
#     ## Description
#     This environment corresponds to the Swimmer environment described in Rémi Coulom's PhD thesis ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document).
#     The environment aims to increase the number of independent state and control variables compared to classical control environments.
#     The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
#     The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
#     and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction.
#
#     ## Notes
#
#     The problem parameters are:
#     Problem parameters:
#     * *n*: number of body parts
#     * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n})
#     * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n})
#     * *k*: viscous-friction coefficient
#
#     While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1, and *k* = 0.1.
#     It is possible to pass a custom MuJoCo XML file during construction to increase the number of links, or to tweak any of the parameters.
#
#
#     ## Action Space
#     ```{figure} action_space_figures/swimmer.png
#     :name: swimmer
#     ```
#
#     The action space is a `Box(-1, 1, (2,), float32)`. An action represents the torques applied between *links*
#
#     | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
#     |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
#     | 0   | Torque applied on the first rotor  | -1          | 1           | motor1_rot                       | hinge | torque (N m) |
#     | 1   | Torque applied on the second rotor | -1          | 1           | motor2_rot                       | hinge | torque (N m) |
#
#
#     ## Observation Space
#     The observation space consists of the following parts (in order):
#
#     - *qpos (3 elements by default):* Position values of the robot's body parts.
#     - *qvel (5 elements):* The velocities of these individual body parts (their derivatives).
#
#     By default, the observation does not include the x- and y-coordinates of the front tip.
#     These can be included by passing `exclude_current_positions_from_observation=False` during construction.
#     In this case, the observation space will be a `Box(-Inf, Inf, (10,), float64)`, where the first two observations are the x- and y-coordinates of the front tip.
#     Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
#
#     By default, however, the observation space is a `Box(-Inf, Inf, (8,), float64)` where the elements are as follows:
#
#     | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
#     | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
#     | 0   | angle of the front tip               | -Inf | Inf | free_body_rot                    | hinge | angle (rad)              |
#     | 1   | angle of the first rotor             | -Inf | Inf | motor1_rot                       | hinge | angle (rad)              |
#     | 2   | angle of the second rotor            | -Inf | Inf | motor2_rot                       | hinge | angle (rad)              |
#     | 3   | velocity of the tip along the x-axis | -Inf | Inf | slider1                          | slide | velocity (m/s)           |
#     | 4   | velocity of the tip along the y-axis | -Inf | Inf | slider2                          | slide | velocity (m/s)           |
#     | 5   | angular velocity of front tip        | -Inf | Inf | free_body_rot                    | hinge | angular velocity (rad/s) |
#     | 6   | angular velocity of first rotor      | -Inf | Inf | motor1_rot                       | hinge | angular velocity (rad/s) |
#     | 7   | angular velocity of second rotor     | -Inf | Inf | motor2_rot                       | hinge | angular velocity (rad/s) |
#     | excluded | position of the tip along the x-axis | -Inf | Inf | slider1                          | slide | position (m)           |
#     | excluded | position of the tip along the y-axis | -Inf | Inf | slider2                          | slide | position (m)           |
#
#
#     ## Rewards
#     The total reward is: ***reward*** *=* *forward_reward - ctrl_cost*.
#
#     - *forward_reward*:
#     A reward for moving forward,
#     this reward would be positive if the Swimmer moves forward (in the positive $x$ direction / in the right direction).
#     $w_{forward} \times \frac{dx}{dt}$, where
#     $dx$ is the displacement of the (front) "tip" ($x_{after-action} - x_{before-action}$),
#     $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is 4),
#     and `frametime` which is $0.01$ - so the default is $dt = 4 \times 0.01 = 0.04$,
#
# ---- quick reference ----------------------------------------------------
# Observation space: Box(-inf, inf, (8,), float64)
# Action space:      Box(-1.0, 1.0, (2,), float32)
# Max episode steps: 1000
#
# YOUR TASK: implement Policy.act() (and optionally update() to learn).
# The scaffold runs out of the box with a RANDOM policy — replace it.
#   python Swimmer_v5.py --episodes 20            # evaluate
#   python Swimmer_v5.py --render                 # watch it play
# ==========================================================================
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: F401  (handy for building policies)
from common import BasePolicy, run, parse_args

ENV_ID = "Swimmer-v5"


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
