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
