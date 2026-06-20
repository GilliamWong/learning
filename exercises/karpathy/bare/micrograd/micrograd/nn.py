# nn.py -- a tiny neural-network library built on the Value autograd engine.
#
# Implement the pieces of a multi-layer perceptron, expressed entirely in terms
# of `Value`s: a base module (exposing parameters() and zero_grad()), a single
# neuron, a layer of neurons, and an MLP (a stack of layers).
#
# Input:    a list of numbers / Values (one feature vector).
# Output:   a Value or list of Values (the forward pass), and parameters() must
#           return every trainable Value so an optimizer can update them.
# Behavior: calling a module runs a forward pass; weights are initialized
#           randomly when the module is constructed.

import random
from micrograd.engine import Value
