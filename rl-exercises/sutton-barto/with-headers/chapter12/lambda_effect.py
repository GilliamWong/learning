# lambda_effect.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The effect of the trace-decay parameter lambda (Chapter 12). Sweep lambda
# across the chapter's tasks/algorithms to show its influence on performance.
#
# Input: none. Output: performance-vs-lambda curves.


#######################################################################
# Copyright (C)                                                       #
# 2021 Johann Huber (huber.joh@hotmail.fr)                            #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

"""

Description:
    This script is meant to reproduce Figure 12.14 of Sutton and Barto's book. This example shows
    the effect of λ on 4 reinforcement learning tasks.

Credits:
    The "Cart and Pole" environment's code has been taken from openai gym source code.
        Link : https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L7
    The tile coding software has been taken from Sutton's website.
        Link : http://www.incompleteideas.net/tiles/tiles3.html

Remark:
    - The optimum step-size parameters search have been omitted to avoid an even longer code. This
    problem has already been met several times in the chapter.


Structure:
    1. Utils
        1.1. Tiling utils
        1.2. Eligibility traces utils
    2. Random walk
    3. Mountain Car
    4. Cart and Pole
    5. Results
        5.1. Getting plot data
        5.2. Reproducing figure 12.14
        5.3. Main

""";


import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()


#############################################################################################
#                                          1. Utils                                         #
#############################################################################################

#-------------------#
# 1.1. Tiling utils #
#-------------------#

# Credit : http://www.incompleteideas.net/tiles/tiles3.html

basehash = hash

class IHT:
    """Structure to handle collisions."""

    def __init__(self, sizeval):
        # TODO: implement IHT.__init__
        raise NotImplementedError

    def __str__(self):
        """Prepares a string for printing whenever this object is printed."""
        # TODO: implement IHT.__str__
        raise NotImplementedError

    def count(self):
        # TODO: implement IHT.count
        raise NotImplementedError

    def fullp(self):
        # TODO: implement IHT.fullp
        raise NotImplementedError

    def getindex(self, obj, readonly=False):
        # TODO: implement IHT.getindex
        raise NotImplementedError

def hashcoords(coordinates, m, readonly=False):
    # TODO: implement hashcoords
    raise NotImplementedError

from math import floor, log
from itertools import zip_longest

def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """Returns num-tilings tile indices corresponding to the floats and ints"""
    # TODO: implement tiles
    raise NotImplementedError


def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """Returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    # TODO: implement tileswrap
    raise NotImplementedError


class IndexHashTable:

    def __init__(self, iht_size, num_tilings, tiling_size, obs_bounds):
        # Index Hash Table size
        # TODO: implement IndexHashTable.__init__
        raise NotImplementedError


    def get_tiles(self, state, action):
        """Get the encoded state_action using Sutton's grid tiling software."""
        # List of floats numbers to be tiled
        # TODO: implement IndexHashTable.get_tiles
        raise NotImplementedError


#-------------------------------#
# 1.2. Eligibility traces utils #
#-------------------------------#


def update_trace_vector(agent, method, state, action=None):
    """Updates agent's trace vector (z) with then current state (or state-action pair) using to the given method.
    Returns the updated vector."""

    # TODO: implement update_trace_vector
    raise NotImplementedError


#############################################################################################
#                                     2. Random walk                                        #
#############################################################################################

class RandomWalkEnvironment:

    def __init__(self):
        # Number of states
        # TODO: implement RandomWalkEnvironment.__init__
        raise NotImplementedError

    def step(self, state, action):
        # TODO: implement RandomWalkEnvironment.step
        raise NotImplementedError


class RandomWalkAgent:
    def __init__(self, lmbda, alpha):
        # Number of states
        # TODO: implement RandomWalkAgent.__init__
        raise NotImplementedError

    @property
    def error_hist(self):
        # TODO: implement RandomWalkAgent.error_hist
        raise NotImplementedError

    def get_all_v_hat(self):
        # TODO: implement RandomWalkAgent.get_all_v_hat
        raise NotImplementedError

    def policy(self, state):
        """Action selection : uniform distribution. State argument is given for consistency."""
        # TODO: implement RandomWalkAgent.policy
        raise NotImplementedError

    def v_hat(self, state):
        """Returns the approximated value for state, w.r.t. the weight vector."""
        # TODO: implement RandomWalkAgent.v_hat
        raise NotImplementedError

    def grad_v_hat(self, state):
        """Compute the gradient of the state value w.r.t. the weight vector."""
        # TODO: implement RandomWalkAgent.grad_v_hat
        raise NotImplementedError

    def get_active_features(self, state):
        """Get an array containing the id of the current active feature."""
        # TODO: implement RandomWalkAgent.get_active_features
        raise NotImplementedError

    def run_td_lambda(self, env, n_episodes, method):
        """Method described p293 of the book.

        :param env: environment to interact with.
        :param n_episodes: number of episodes to train on.
        :param method: specify the TD(λ) method :
                * 'accumulating' : With accumulating traces ;
                * 'replace' : With replacing traces ;
        :return: None
        """

        # TODO: implement RandomWalkAgent.run_td_lambda
        raise NotImplementedError


class RandomWalk:
    def __init__(self, lmbda, alpha):
        # TODO: implement RandomWalk.__init__
        raise NotImplementedError

    @property
    def error_hist(self):
        # TODO: implement RandomWalk.error_hist
        raise NotImplementedError

    def train(self, n_episodes, method):
        # TODO: implement RandomWalk.train
        raise NotImplementedError


#############################################################################################
#                                     3. Mountain Car                                       #
#############################################################################################

class MountainCarEnvironment:

    def __init__(self):
        # Action space
        # TODO: implement MountainCarEnvironment.__init__
        raise NotImplementedError

    def step(self, state, action):
        # TODO: implement MountainCarEnvironment.step
        raise NotImplementedError


class MountainCarAgent:
    def __init__(self, alpha, lmbda, iht_args):
        # Index Hash Table for position encoding
        # TODO: implement MountainCarAgent.__init__
        raise NotImplementedError

    @property
    def n_step_hist(self):
        # TODO: implement MountainCarAgent.n_step_hist
        raise NotImplementedError

    def policy(self, state):
        """Apply a ε-greedy policy to choose an action from state."""

        # Always greedy : exploration is assured by optimistic initial values
        # TODO: implement MountainCarAgent.policy
        raise NotImplementedError

    def get_init_state(self):
        """Get a random starting position in the interval [-0.6, -0.4)."""
        # TODO: implement MountainCarAgent.get_init_state
        raise NotImplementedError

    def is_terminal_state(self, state):
        # TODO: implement MountainCarAgent.is_terminal_state
        raise NotImplementedError

    def q_hat(self, state, action):
        """Compute the q value for the current state-action pair."""
        # TODO: implement MountainCarAgent.q_hat
        raise NotImplementedError

    def get_active_features(self, state, action):
        """Get an array containing the ids of the current active features."""
        # TODO: implement MountainCarAgent.get_active_features
        raise NotImplementedError

    def run_sarsa_lambda(self, env, n_episodes, method):
        """Apply Sarsa(λ) algorithm. (p.305)

        :param env: environment to interact with.
        :param n_episodes: number of episodes to train on.
        :param method: specify the Sarsa(λ) method :
                * 'accumulating' : With accumulating traces ;
                * 'replace' : With replacing traces ;
        :return: None
        """

        # TODO: implement MountainCarAgent.run_sarsa_lambda
        raise NotImplementedError


class MountainCar:
    def __init__(self, lmbda, alpha):
        # Environment initialization
        # TODO: implement MountainCar.__init__
        raise NotImplementedError

    @property
    def n_step_hist(self):
        # TODO: implement MountainCar.n_step_hist
        raise NotImplementedError

    def train(self, n_episodes, method):
        # TODO: implement MountainCar.train
        raise NotImplementedError


#############################################################################################
#                                     4. Cart and Pole                                      #
#############################################################################################

class CartPoleEnvironment:
    """Credit : https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L7"""

    def __init__(self):

        # TODO: implement CartPoleEnvironment.__init__
        raise NotImplementedError

    def is_state_valid(self, state):
        # TODO: implement CartPoleEnvironment.is_state_valid
        raise NotImplementedError

    def step(self, state, action):
        # TODO: implement CartPoleEnvironment.step
        raise NotImplementedError


class CartPoleAgent:
    def __init__(self, iht_args, alpha, lmbda):
        # Index Hash Table for position encoding
        # TODO: implement CartPoleAgent.__init__
        raise NotImplementedError

    @property
    def n_failures(self):
        # TODO: implement CartPoleAgent.n_failures
        raise NotImplementedError

    def policy(self, state):
        """Apply a ε-greedy policy to choose an action from state."""
        # TODO: implement CartPoleAgent.policy
        raise NotImplementedError

    def is_state_valid(self, state):
        # TODO: implement CartPoleAgent.is_state_valid
        raise NotImplementedError

    def get_init_state(self):
        """Get a random starting position."""
        # TODO: implement CartPoleAgent.get_init_state
        raise NotImplementedError

    def is_state_over_bounds(self, state):
        """Returns True if the current state is out of bounds, i.e. the current run is over. Returns
        False otherwise."""

        # TODO: implement CartPoleAgent.is_state_over_bounds
        raise NotImplementedError

    def q_hat(self, state, action):
        """Compute the q value for the current state-action pair."""
        # TODO: implement CartPoleAgent.q_hat
        raise NotImplementedError

    def get_active_features(self, state, action):
        """Get an array containing the ids of the current active features."""
        # TODO: implement CartPoleAgent.get_active_features
        raise NotImplementedError

    def run_sarsa_lambda(self, env, n_step_max, method):
        """Apply Sarsa(λ) algorithm. (p.305)

        :param env: environment to interact with.
        :param n_step_max: number of steps to train on.
        :param method: specify the Sarsa(λ) method :
                * 'accumulating' : With accumulating traces ;
        :return: None
        """
        # TODO: implement CartPoleAgent.run_sarsa_lambda
        raise NotImplementedError

        #print('Running over. n_ep =', n_ep)


class CartPole:
    def __init__(self, lmbda, alpha):
        # Environment initialization
        # TODO: implement CartPole.__init__
        raise NotImplementedError

    @property
    def n_failures(self):
        # TODO: implement CartPole.n_failures
        raise NotImplementedError

    def train(self, n_step_max, method):
        # TODO: implement CartPole.train
        raise NotImplementedError



#############################################################################################
#                                     5. Puddle World                                       #
#############################################################################################

class PuddleWorldGrid:
    def __init__(self):
        # Grid dimensions
        # TODO: implement PuddleWorldGrid.__init__
        raise NotImplementedError

    @property
    def height(self):
        # TODO: implement PuddleWorldGrid.height
        raise NotImplementedError

    @property
    def width(self):
        # TODO: implement PuddleWorldGrid.width
        raise NotImplementedError

    def is_state_goal(self, state):
        # TODO: implement PuddleWorldGrid.is_state_goal
        raise NotImplementedError

    def get_dist2puddle(self, state):
        """Get state's distance (float) to the nearest puddle's border.
        Returns a float corresponding to the state's distance to the nearest puddle border. Return -1 if
        the state to evaluate is far enough from puddles to be not affected by the cost penalty.
        """
        # TODO: implement PuddleWorldGrid.get_dist2puddle
        raise NotImplementedError

    def cvt_ij2xy(self, pos_ij):
        # TODO: implement PuddleWorldGrid.cvt_ij2xy
        raise NotImplementedError

    def draw(self):
        # TODO: implement PuddleWorldGrid.draw
        raise NotImplementedError


class PuddleWorldEnvironment:
    def __init__(self, grid):
        # Grid object
        # TODO: implement PuddleWorldEnvironment.__init__
        raise NotImplementedError

    def step(self, state, action):
        # Random gaussian noise (std=0.01) on each move
        # TODO: implement PuddleWorldEnvironment.step
        raise NotImplementedError


class PuddleWorldAgent:
    def __init__(self, grid, alpha, lmbda, iht_args):
        # Index Hash Table for position encoding
        # TODO: implement PuddleWorldAgent.__init__
        raise NotImplementedError

    @property
    def cost_per_ep_hist(self):
        # TODO: implement PuddleWorldAgent.cost_per_ep_hist
        raise NotImplementedError

    def policy(self, state):
        """Apply a ε-greedy policy to choose an action from state."""

        # TODO: implement PuddleWorldAgent.policy
        raise NotImplementedError

    def get_start_pos(self):
        """Randomly pick a non-goal state as starting position."""
        # TODO: implement PuddleWorldAgent.get_start_pos
        raise NotImplementedError

    def is_terminal_state(self, state):
        # TODO: implement PuddleWorldAgent.is_terminal_state
        raise NotImplementedError

    def q_hat(self, state, action):
        """Compute the q value for the current state-action pair."""
        # TODO: implement PuddleWorldAgent.q_hat
        raise NotImplementedError

    def get_active_features(self, state, action):
        """Get an array containing the ids of the current active features."""
        # TODO: implement PuddleWorldAgent.get_active_features
        raise NotImplementedError

    def run_sarsa_lambda(self, env, n_episodes, method):
        """Apply Sarsa(λ) algorithm. (p.305)

        :param env: environment to interact with.
        :param n_episodes: number of episodes to train on.
        :param method: specify the Sarsa(λ) method :
                * 'accumulating' : With accumulating traces ;
                * 'replace' : With replacing traces ;
                * 'replace_reset' : With replacing traces, and clearing the traces of other actions.
        :return: None
        """
        # TODO: implement PuddleWorldAgent.run_sarsa_lambda
        raise NotImplementedError


class PuddleWorld:
    def __init__(self, lmbda, alpha):
        # Grid initialization
        # TODO: implement PuddleWorld.__init__
        raise NotImplementedError

    @property
    def cost_per_ep_hist(self):
        # TODO: implement PuddleWorld.cost_per_ep_hist
        raise NotImplementedError

    def draw(self):
        # TODO: implement PuddleWorld.draw
        raise NotImplementedError

    def train(self, n_episodes, method):
        # TODO: implement PuddleWorld.train
        raise NotImplementedError


def get_puddle_world_map():
    """Creates the puddle world map and save the figure in the local folder as a .png file."""
    # TODO: implement get_puddle_world_map
    raise NotImplementedError


#############################################################################################
#                                        5. Results                                         #
#############################################################################################

#---------------------------#
# 5.1. Getting plot data    #
#---------------------------#


def get_random_walk_plot_data():
    # TODO: implement get_random_walk_plot_data
    raise NotImplementedError

def get_mountain_car_plot_data():
    # TODO: implement get_mountain_car_plot_data
    raise NotImplementedError


def get_cart_pole_plot_data():
    # TODO: implement get_cart_pole_plot_data
    raise NotImplementedError


def get_puddle_world_plot_data():
    # TODO: implement get_puddle_world_plot_data
    raise NotImplementedError


#----------------------------------#
# 5.2. Reproducing figure 12.14    #
#----------------------------------#

def figure_12_14():
    # Get plot data for each task
    # TODO: implement figure_12_14
    raise NotImplementedError
    #plt.waitforbuttonpress()

#--------------#
# 5.3. Main    #
#--------------#

if __name__ == '__main__':

    figure_12_14() # ~2h on colab

    #get_puddle_world_map()


