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

def compute_fp(param):
    a = param[0]
    b = param[1]
    fp = np.array([-400*a*b + 400*a**3 + 2*a - 2, 200*b - 200*a**2])
    return fp

# Adam optimization
def adam_optimization(x, y, ax):
    learning_rate = 0.5
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    num_iterations = 300

    # Initial parameters
    param = np.array([-2.0, 2.0])

    # Initialize moments
    m = np.zeros_like(param)
    v = np.zeros_like(param)

    # Create a text object for loss display
    bkg = dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    loss_text = ax.text(-1, -2, "", fontsize=18*gscale, color="black", bbox=bkg)

    for iter in range(1, num_iterations + 1):
        fp = compute_fp(param)

        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * fp
        v = beta2 * v + (1 - beta2) * (fp ** 2)

        # Correct bias in moments
        m_hat = m / (1 - beta1 ** iter)
        v_hat = v / (1 - beta2 ** iter)

        # Update parameters
        param = param - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Update loss display
        loss = compute_f(param[0], param[1])
        loss_text.set_text(f"Iter={iter}: loss = {loss:0.3f}")

        # Plotting current param
        ax.scatter(param[0], param[1], color='red', s=10*gscale)
        plt.pause(0.1)

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
plt.title('Adam Optimization')
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.contourf(X, Y, Z, levels=256, cmap='jet')
ax.contour(X, Y, Z, levels=128, colors='k', linewidths=1, linestyles='-')

# Run Adam optimization
param = adam_optimization(x, y, ax)
plt.show()
