# maze.py -- Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)
#
# The Dyna maze (Chapter 8, Figures 8.2/8.4/8.5, Example 8.4). Model-based RL:
# Dyna-Q and Dyna-Q+ with planning, plus prioritized sweeping, on gridworld
# mazes (including blocking/shortcut changes).
#
# Input: none. Output: learning curves vs number of planning steps.


#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy

class PriorityQueue:
    def __init__(self):
        # TODO: implement PriorityQueue.__init__
        raise NotImplementedError

    def add_item(self, item, priority=0):
        # TODO: implement PriorityQueue.add_item
        raise NotImplementedError

    def remove_item(self, item):
        # TODO: implement PriorityQueue.remove_item
        raise NotImplementedError

    def pop_item(self):
        # TODO: implement PriorityQueue.pop_item
        raise NotImplementedError

    def empty(self):
        # TODO: implement PriorityQueue.empty
        raise NotImplementedError

# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:
    def __init__(self):
        # maze width
        # TODO: implement Maze.__init__
        raise NotImplementedError

    # extend a state to a higher resolution maze
    # @state: state in lower resolution maze
    # @factor: extension factor, one state will become factor^2 states after extension
    def extend_state(self, state, factor):
        # TODO: implement Maze.extend_state
        raise NotImplementedError

    # extend a state into higher resolution
    # one state in original maze will become @factor^2 states in @return new maze
    def extend_maze(self, factor):
        # TODO: implement Maze.extend_maze
        raise NotImplementedError

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
        # TODO: implement Maze.step
        raise NotImplementedError

# a wrapper class for parameters of dyna algorithms
class DynaParams:
    def __init__(self):
        # discount
        # TODO: implement DynaParams.__init__
        raise NotImplementedError


# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze, dyna_params):
    # TODO: implement choose_action
    raise NotImplementedError

# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        # TODO: implement TrivialModel.__init__
        raise NotImplementedError

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        # TODO: implement TrivialModel.feed
        raise NotImplementedError

    # randomly sample from previous experience
    def sample(self):
        # TODO: implement TrivialModel.sample
        raise NotImplementedError

# Time-based model for planning in Dyna-Q+
class TimeModel:
    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, maze, time_weight=1e-4, rand=np.random):
        # TODO: implement TimeModel.__init__
        raise NotImplementedError

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        # TODO: implement TimeModel.feed
        raise NotImplementedError

    # randomly sample from previous experience
    def sample(self):
        # TODO: implement TimeModel.sample
        raise NotImplementedError

# Model containing a priority queue for Prioritized Sweeping
class PriorityModel(TrivialModel):
    def __init__(self, rand=np.random):
        # TODO: implement PriorityModel.__init__
        raise NotImplementedError

    # add a @state-@action pair into the priority queue with priority @priority
    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        # TODO: implement PriorityModel.insert
        raise NotImplementedError

    # @return: whether the priority queue is empty
    def empty(self):
        # TODO: implement PriorityModel.empty
        raise NotImplementedError

    # get the first item in the priority queue
    def sample(self):
        # TODO: implement PriorityModel.sample
        raise NotImplementedError

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        # TODO: implement PriorityModel.feed
        raise NotImplementedError

    # get all seen predecessors of a state @state
    def predecessor(self, state):
        # TODO: implement PriorityModel.predecessor
        raise NotImplementedError


# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
def dyna_q(q_value, model, maze, dyna_params):
    # TODO: implement dyna_q
    raise NotImplementedError

# play for an episode for prioritized sweeping algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm
# @return: # of backups during this episode
def prioritized_sweeping(q_value, model, maze, dyna_params):
    # TODO: implement prioritized_sweeping
    raise NotImplementedError

# Figure 8.2, DynaMaze, use 10 runs instead of 30 runs
def figure_8_2():
    # set up an instance for DynaMaze
    # TODO: implement figure_8_2
    raise NotImplementedError

# wrapper function for changing maze
# @maze: a maze instance
# @dynaParams: several parameters for dyna algorithms
def changing_maze(maze, dyna_params):

    # set up max steps
    # TODO: implement changing_maze
    raise NotImplementedError

# Figure 8.4, BlockingMaze
def figure_8_4():
    # set up a blocking maze instance
    # TODO: implement figure_8_4
    raise NotImplementedError

# Figure 8.5, ShortcutMaze
def figure_8_5():
    # set up a shortcut maze instance
    # TODO: implement figure_8_5
    raise NotImplementedError

# Check whether state-action values are already optimal
def check_path(q_values, maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    # TODO: implement check_path
    raise NotImplementedError

# Example 8.4, mazes with different resolution
def example_8_4():
    # get the original 6 * 9 maze
    # TODO: implement example_8_4
    raise NotImplementedError

if __name__ == '__main__':
    figure_8_2()
    figure_8_4()
    figure_8_5()
    example_8_4()

