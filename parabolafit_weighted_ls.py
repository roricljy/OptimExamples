import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

# Generate data
x = np.arange(1, 101)
y = -2 * (x - 40)**2 + 30 + 100 * np.random.randn(100)
y[49:70] += 500 * np.abs(np.arange(50, 71) - 60) - 5000

# Create matrix A
A = np.vstack([x**2, x, np.ones(len(x))]).T

# Initialize the solution (Least Squares method)
p = np.linalg.pinv(A) @ y

# Iterative algorithm
for itr in range(15):
    r = A @ p - y
    W = np.diag(1 / (1 + np.abs(r) / 1.3998))
    p = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ y)
    
    # Plot the data and the fitted curve
    bkg=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    plt.text(0.35, 0.5, f"Iteration {itr}", transform=plt.gca().transAxes, fontsize=20, color="black", bbox=bkg)
    plt.plot(x, y, '*b')
    plt.plot(x, A @ p, 'r', linewidth=4)
    plt.pause(0.5)
    plt.clf()

# Plot the final result
plt.plot(x, y, '*b')
plt.plot(x, A @ p, 'r', linewidth=4)
plt.show()