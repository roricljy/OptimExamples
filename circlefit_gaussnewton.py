import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams.update({'font.size': 14})

# Gauss-Newton method to fit a circle: (x - a)^2 + (y - b)^2 = c^2
params_list = []
def gauss_newton_progress(x, y, initial_guess, max_iter=20):
    params = initial_guess
    for _ in range(max_iter):
        params_list.append(params.copy())
        a, b, c = params
        res = np.sqrt((x - a)**2 + (y - b)**2) - c
        J = np.vstack([(a - x) / np.sqrt((x - a)**2 + (y - b)**2), 
                       (b - y) / np.sqrt((y - b)**2 + (x - a)**2), 
                       -np.ones(len(x))]).T    
        delta = np.linalg.lstsq(J, res, rcond=None)[0]
        #delta = np.linalg.inv(J.T @ J + np.eye(3)) @ J.T @ res
        params -= delta
        if np.linalg.norm(delta) < 1e-6:
            break
    params_list.append(params)
    return params

# Generate data points (example data)
current_time_seed = int(time.time())
np.random.seed(current_time_seed)
actual_center = np.array([5, 5])
actual_radius = 3
angles = np.linspace(0, 2 * np.pi, 10)
data_points = actual_center + actual_radius * np.column_stack((np.cos(angles), np.sin(angles)))
data_points += np.random.normal(scale=0.3, size=data_points.shape)  # Add noise
x_data, y_data = data_points[:, 0], data_points[:, 1]

# Initial guess
#initial_guess = np.array([3., 3., 2.])
initial_guess = np.array([1., 1., 1.])

# Circle fitting
estimated_params = gauss_newton_progress(x_data, y_data, initial_guess)
xc_est, yc_est, r_est = estimated_params

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x_data, y_data, label='Data Points')
circle_line, = ax.plot([], [], 'r-', label='Estimated Circle')
point, = ax.plot([], [], 'ro')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.axis('equal')
ax.set_title('Circle Fitting using Gauss-Newton Method')

theta = np.linspace(0, 2 * np.pi, 100)

def init():
    circle_line.set_data([], [])
    point.set_data([], [])
    return circle_line, point

def update(frame):
    xc, yc, r = params_list[frame]
    estimated_circle = np.column_stack((xc + r * np.cos(theta), yc + r * np.sin(theta)))
    circle_line.set_data(estimated_circle[:, 0], estimated_circle[:, 1])
    point.set_data([xc], [yc])
    return circle_line, point

ani = FuncAnimation(fig, update, frames=len(params_list), init_func=init, blit=True, repeat=False)
plt.show()