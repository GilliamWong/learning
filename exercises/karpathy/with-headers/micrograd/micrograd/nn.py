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

class Module:

    def zero_grad(self):
        # TODO: implement Module.zero_grad
        raise NotImplementedError

    def parameters(self):
        # TODO: implement Module.parameters
        raise NotImplementedError

class Neuron(Module):

    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        #python uses this function with n(x) when n is a neuron
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    #n_in defines the dim of each neuron, and n_out defines the number of neurons. 
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):

    #now nin is the number of neurons in the input layer and nouts is now a list defining the sizes of each layer and how many there are. 
    def __init__(self, n_in, n_outs):
        #sizes holds the initial input dim, then the output dims for each layer. So when we do the self.layers line then sizes[i] is the input dim of each layer (the output dim of the prev) and sizes[i + 1] holds the number from n_outs
        sizes = [n_in] + n_outs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]
