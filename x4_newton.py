import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Define the function and its derivative
def f(x):
    return x**4

def fp(x):
    return 4 * x**3

def fpp(x):
    return 12 * x**2
    
# Configurations
class Config:
    term_max_iter = 100
    term_tolerance = 1e-6

config = Config()

# Initialize x
x = 2
x_values = [x]  # Store x values for plotting

# Setting up the plot for intermediate steps
x_plot = np.linspace(-2, 2, 400)
y_plot = f(x_plot)

fig, ax = plt.subplots(figsize=(10*gscale, 7*gscale))
ax.plot(x_plot, y_plot, label='y = x^4', linewidth=3*gscale, color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Newton\'s method: y = x^4')

# Intermediate steps plotted
for itr in range(config.term_max_iter):
    x_new = x - fp(x)/fpp(x)
    x_values.append(x_new)
    
    # Plot the current step
    bkg=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    ax.text(0.4, 0.5, f"Iteration {itr}", transform=ax.transAxes, fontsize=20*gscale, color="black", bbox=bkg)
    ax.scatter(x_values, [f(x) for x in x_values], color='red', marker='x', s=100*gscale)
    plt.pause(0.1)
    
    # Check the tolerance terminal condition
    if np.linalg.norm(x_new - x) < config.term_tolerance:
        break
    
    x = x_new

# Final legend and show plot
plt.show()