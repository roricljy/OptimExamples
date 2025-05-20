import numpy as np
import time

# Settings
learning_rate = 0.02
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
    x_new = x - learning_rate * fp
    print(f"iter{iter}: loss = {f}")
    if abs(x_new - x) < tolerance:
        break
    x = x_new
    time.sleep(0.05)

# Print the result
print(f"The root is {x}")
print(f"model({x}) = {model(x)}")