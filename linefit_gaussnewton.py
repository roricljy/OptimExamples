import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Function to compute Jacobian of the residual vector
# ri = |axi + byi + c| / sqrt(a^2 + b^2)
def compute_Jacobian(x, y, a, b, c):    
    # Compute the denominator term for normalization
    residual = a * x + b * y + c
    sign_residual = np.sign(residual)    
    denom = np.sqrt(a**2 + b**2)
    denom_cubed = denom**3
    
    # Initialize the Jacobian matrix
    n = len(x)
    jacobian = np.zeros((n, 3))
    
    # Fill the Jacobian matrix
    jacobian[:, 0] = (sign_residual * x * (a**2 + b**2) - np.abs(residual) * a) / denom_cubed
    jacobian[:, 1] = (sign_residual * y * (a**2 + b**2) - np.abs(residual) * b) / denom_cubed
    jacobian[:, 2] = sign_residual / denom                                      # Partial derivative w.r.t. c
    
    return jacobian
    
# Newton's method to fit a line ax + by + c = 0
def newton_method(x, y, learning_rate=0.9, num_iterations=50):
    # Initial parameters a, b, c
    a = 1.0
    b = 1.0
    c = 1.0

    font_bkg=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    for iter in range(num_iterations):
        plt.clf()

        # Update parameters using Gauss-Newton method
        res = np.abs(x * a + y * b + c) / np.sqrt(a**2 + b**2)
        J = compute_Jacobian(x, y, a, b, c)
        delta = np.linalg.lstsq(J, res, rcond=None)[0] # Gauss-Newton

        a -= learning_rate * delta[0]
        b -= learning_rate * delta[1]
        c -= learning_rate * delta[2]

        # Print loss
        loss = np.sum((a*x + b*y + c)**2/(a**2 + b**2))        
        plt.text(0.3, 0.3, f"Iter={iter+1}: loss = {loss:0.3f}", transform=plt.gca().transAxes, fontsize=18*gscale, color="black", bbox=font_bkg)            

        # Plotting the data points and the fitted line
        plt.scatter(x, y, color='blue', s=100*gscale, label='Data points')

        # Calculate fitted line points
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = -(a * x_vals + c) / b

        plt.plot(x_vals, y_vals, color='red', linewidth=3*gscale, label='Fitted line')
        plt.xlim(0, 8)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Line Fitting: Gauss-Newton method')
        plt.pause(0.1)

    return a, b, c

# Example usage
# Sample data points
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2, 2, 3, 5, 4, 7, 7])

# Init display
plt.figure(figsize=(7*gscale, 7*gscale))
plt.pause(0.1)

# Run gradient descent
a, b, c = newton_method(x, y)
plt.show()
print(f"Fitted line parameters: a = {a}, b = {b}, c = {c}")