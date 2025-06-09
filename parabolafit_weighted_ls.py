import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Generate data
x = np.arange(1, 101)
y = -2 * (x - 40)**2 + 30 + 100 * np.random.randn(100)
y[50:80] += 400 * np.abs(np.arange(50, 80) - 65) - 7000

# Create matrix A
A = np.vstack([x**2, x, np.ones(len(x))]).T

# Initialize the solution (Least Squares method)
p = np.linalg.pinv(A) @ y

# Init display
plt.figure(figsize=(7*gscale, 7*gscale))
plt.pause(0.1)

# Iterative algorithm
for itr in range(100):
    r = A @ p - y
    W = np.diag(1 / (1 + np.abs(r) / 1.3998))
    p = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ y)
    
    # Plot the data and the fitted curve
    bkg=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    plt.text(0.35, 0.5, f"Iteration {itr}", transform=plt.gca().transAxes, fontsize=20, color="black", bbox=bkg)
    plt.plot(x, y, '*b', markersize=6*gscale)
    plt.plot(x, A @ p, 'r', linewidth=3*gscale)
    plt.pause(0.2)
    plt.clf()

# Plot the final result
plt.plot(x, y, '*b', markersize=6*gscale)
plt.plot(x, A @ p, 'r', linewidth=3*gscale)
plt.show()