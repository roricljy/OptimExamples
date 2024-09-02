import numpy as np
import matplotlib.pyplot as plt

# Function to compute gradient and Hessian of the objective function
# J = sum{(ax + by + c)^2/(a^2 + b^2)}
def compute_gradient_hessian(x, y, a, b, c):
    m = len(x)

    # Compute the common numerator r and denominator d
    r = a*x + b*y + c
    d = a**2 + b**2

    # Compute the gradients
    da = (2/m) * np.sum(r * (x * d - a * r) / d**2)
    db = (2/m) * np.sum(r * (y * d - b * r) / d**2)
    dc = (2/m) * np.sum(r / d)
    G = np.array([da, db, dc])

    # Compute the Hessian    
    H = np.zeros((3, 3))
    H[0, 0] = (2/m) * np.sum((x**2)/d - (r**2)/d**2 + 4*(a**2)*(r**2)/d**3 - (4*a*x*r)/d**2)
    H[0, 1] = H[1, 0] = (2/m) * np.sum((x*y)/d - (2*a*y*r)/d**2 - (2*b*x*r)/d**2 + 4*a*b*(r**2)/d**3)
    H[0, 2] = H[2, 0] = (2/m) * np.sum(-(x*a**2 + 2*y*a*b + 2*c*a - x*b**2)/d**2)
    H[1, 1] = (2/m) * np.sum((y**2)/d - (r**2)/d**2 + (4*b**2)*(r**2)/d**3 - (4*b*y*r)/d**2)
    H[1, 2] = H[2, 1] = (2/m) * np.sum(-(- y*a**2 + 2*x*a*b + y*b**2 + 2*c*b)/d**2)
    H[2, 2] = (2/m) * np.sum(1/d)

    return G, H

# Newton's method to fit a line ax + by + c = 0
def newton_method(x, y, learning_rate=0.9, num_iterations=100):
    # Initial parameters a, b, c
    a = 1.0
    b = 1.0
    c = 1.0

    for iter in range(num_iterations):
        gradients, H = compute_gradient_hessian(x, y, a, b, c)

        # Update parameters using Newton's method
        eigvals, eigvecs = np.linalg.eigh(H)
        H_abs = eigvecs @ np.diag(np.abs(eigvals)) @ eigvecs.T
        delta = np.linalg.solve(H, gradients)                      # Newton's method
        #delta = np.linalg.solve(H_abs, gradients)                 # Saddle-free Newton
        #delta = np.linalg.solve(H_abs + np.eye(3) * 0.01, gradients)  # Saddle-free + Damping

        a -= learning_rate * delta[0]
        b -= learning_rate * delta[1]
        c -= learning_rate * delta[2]

        # Print loss
        loss = np.sum((a*x + b*y + c)**2/(a**2 + b**2))
        print(f"Iter={iter}: loss = {loss}")

        # Plotting the data points and the fitted line
        plt.scatter(x, y, color='blue', label='Data points')

        # Calculate fitted line points
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = -(a * x_vals + c) / b

        plt.plot(x_vals, y_vals, color='red', label='Fitted line')
        plt.xlim(0, 6)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Line Fitting using Gradient Descent')
        plt.pause(0.1)
        plt.clf()

    return a, b, c

# Example usage
# Sample data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2, 3, 5, 4])

# Run gradient descent
a, b, c = newton_method(x, y)

print(f"Fitted line parameters: a = {a}, b = {b}, c = {c}")

# Plotting the data points and the fitted line
plt.scatter(x, y, color='blue', label='Data points')

# Calculate fitted line points
x_vals = np.linspace(min(x), max(x))
y_vals = -(a * x_vals + c) / b

plt.plot(x_vals, y_vals, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Line Fitting using Newton\'s Method')
plt.show()