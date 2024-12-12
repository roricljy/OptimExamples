import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# function to minimize: Rosenbrock function
# : minimum at f(1,1) = 0 in case of n = 2
def compute_f(x, y):
    f = 100*(y - x**2)**2 + (1 - x)**2
    return f
    
def compute_fp(param):
    a = param[0]
    b = param[1]
    fp = np.array([-400*a*b + 400*a**3 + 2*a - 2, 200*b - 200*a**2])
    return fp

# Gradient descent
def gradient_descent(x, y, ax):
    learning_rate = 0.0005
    gamma = 0.0 # momentum
    num_iterations=300

    # Initial parameters
    param = np.array([-2.0, 2.0])    

    # Create a text object for loss display
    bkg = dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    loss_text = ax.text(-1, -2, "", fontsize=18*gscale, color="black", bbox=bkg)

    delta_prev = np.array([0, 0])
    for iter in range(num_iterations):
        fp = compute_fp(param)
        param_prev = param
        param = param - learning_rate * fp + gamma * delta_prev        
        delta_prev = param - param_prev

        # Update loss display
        loss = compute_f(param[0], param[1])
        loss_text.set_text(f"Iter={iter}: loss = {loss:0.3f}")
        
        # Plotting current param
        ax.scatter(param[0], param[1], color='red', s=10*gscale)
        plt.pause(0.1)
           
    return param

# Init display
size = 100
x = np.linspace(-3, 3 , size)
y = np.linspace(-3, 3, size)
X, Y = np.meshgrid(x, y)
Z = compute_f(X, Y)

fig = plt.figure(figsize=(7*gscale, 7*gscale))
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('SGD + Momentum')
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.contourf(X, Y, Z, levels=256, cmap='jet')
ax.contour(X, Y, Z, levels=128, colors='k', linewidths=1, linestyles='-') 

# Run gradient descent
param = gradient_descent(x, y, ax)
plt.show()