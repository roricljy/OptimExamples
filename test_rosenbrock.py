import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Function to minimize: Rosenbrock function
# Minimum at f(1,1) = 0 in case of n = 2
def compute_f(x, y):
    f = 100*(y - x**2)**2 + (1 - x)**2
    return f

def compute_fp(x, y):
    fp = np.array([-400*x*y + 400*x**3 + 2*x - 2, 200*y - 200*x**2])
    return fp

def compute_fpp(x, y):
    fpp = np.array([
        [-400*y + 1200*x**2 + 2, -400*x],
        [-400*x, 200]
    ])
    return fpp    

# optimization
def run_optimization(ax):
    learning_rate = 0.001
    num_iterations = 300
    term_tolerance = 1e-10

    # Initial parameters
    param = np.array([-2.0, 2.0])

    # Create a text object for loss display
    bkg = dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    loss_text = ax.text(-1, -2, "", fontsize=18*gscale, color="black", bbox=bkg)
    ax.scatter(param[0], param[1], color='red', s=10*gscale)
    plt.pause(0.1)
    
    for iter in range(1, num_iterations + 1):
        fp = compute_fp(param[0], param[1])
        fpp = compute_fpp(param[0], param[1])

        # Update parameters
        delta = fp
               
        param = param - learning_rate * delta

        # Update loss display
        loss = compute_f(param[0], param[1])
        loss_text.set_text(f"Iter={iter}: loss = {loss:0.3f}")

        # Plotting current param
        ax.scatter(param[0], param[1], color='red', s=10*gscale)
        plt.pause(0.01)

        # Check convergence
        if np.linalg.norm(delta) < term_tolerance:
                break

    return param

# Init display
size = 100
x = np.linspace(-3, 3, size)
y = np.linspace(-3, 3, size)
X, Y = np.meshgrid(x, y)
Z = compute_f(X, Y)

fig = plt.figure(figsize=(7*gscale, 7*gscale))
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization')
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.contourf(X, Y, Z, levels=256, cmap='jet')
ax.contour(X, Y, Z, levels=128, colors='k', linewidths=1, linestyles='-')

# Run optimization
param = run_optimization(ax)
plt.show()
