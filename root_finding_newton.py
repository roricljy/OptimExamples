import numpy as np

# Settings
tolerance = 1e-12
max_iter = 1000

# Model prediction
def model(x):
    return x**3 + x**2 + x + 1

# Gradient Descent Search
x = 0
for iter in range(max_iter):
    residual = model(x) - 0
    f = residual**2
    fp = 2*residual*(3*x**2 + 2*x + 1)
    fpp = 2*(3*x**2 + 2*x + 1)**2 + 2*residual*(6*x + 2)
    x_new = x - fp / fpp
    print(f"iter{iter}: loss = {f}")
    if abs(x_new - x) < tolerance:
        break
    x = x_new

# Print the result
print(f"The root is {x}")
print(f"model({x}) = {model(x)}")