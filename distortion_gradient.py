import numpy as np

# Settings (example)
learning_rate = 0.01
tolerance = 1e-6
max_iter = 1000
k1 = 0.01
k2 = -0.07

# Brown's distortion model
def distort_function(x):
    return x + k1*x**3 + k2*x**5

# Gradient Descent Search
rd = 0.534
x = rd
for _ in range(max_iter):
    residual = distort_function(x) - rd
    fp = 2*residual*(1 + 3*k1*x**2 + 5*k2*x**4)
    x_new = x - learning_rate * fp
    if abs(x_new - x) < tolerance:
        break
    x = x_new

# Print the result
print(f"The root is {x}.")