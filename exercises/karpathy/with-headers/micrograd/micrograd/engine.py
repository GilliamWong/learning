# engine.py -- a minimal scalar-valued automatic differentiation engine.
#
# Implement a `Value` type that wraps a single number and remembers how it was
# computed, so gradients can be propagated backwards through the whole
# expression.
#
# Input:    Python numbers combined with arithmetic operators.
# Output:   `Value` objects supporting + - * / ** , unary negation and a ReLU,
#           each exposing `.data` and `.grad`, plus a `.backward()`.
# Behavior: composing operations builds an expression graph; calling
#           `.backward()` on the final Value fills in `.grad` for every Value
#           that fed into it (reverse-mode autodiff).
import math

class Value:
    """ stores a single scalar value and its gradient """

    #children is a bit unintuitive, but if you think about it in terms of backprop it makes sense, you're going backwards through
    #the comp graph so the children of each node are the nodes that created it

    #for the backward functions, python remembers the original names and values of vars when you pass it inside a nested function
    #and so for add, backward passes self and other into the result
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        #Assume result has its own gradient. When we call backward on it, then the job of backward is to pass along the gradient of the result 
        #and into the inputs of the add function. It's one step of the entire backward pass implemented at the operator level. 
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        #f = xy. So, df/dx = y, and df/dy = x. Assuming f has some gradient we just multiply it in
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += (out.data * out.grad)
        
        out._backward = _backward
        return out

    def relu(self):
        self.data = max(0, self.data)
        return self.data
    
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += out.grad * (1 - t ** 2)

        out._backward = _backward
        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return self - other

    def __rmul__(self, other): # other * seloh
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward()
        return out

    def __truediv__(self, other): # self / other
        return self * (other ** -1)

    def __rtruediv__(self, other): # other / self
        # TODO: implement Value.__rtruediv__
        raise NotImplementedError
    
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
            topo.append(node)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data})"
