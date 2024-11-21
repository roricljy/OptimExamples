import numpy as np
import matplotlib.pyplot as plt

# Gradient descent function to fit a line ax + by + c = 0
def gradient_descent(x, y, learning_rate=0.01, num_iterations=100):
    # Initial parameters a, b, c
    a = 1.0
    b = 1.0
    c = 1.0

    for iter in range(num_iterations):
        # Compute the common denominator d
        d = (a**2 + b**2)
        
        # Compute the gradients
        da = 2 * np.sum((a*x + b*y + c) * (x * d - a * (a*x + b*y + c)) / d**2)
        db = 2 * np.sum((a*x + b*y + c) * (y * d - b * (a*x + b*y + c)) / d**2)
        dc = 2 * np.sum((a * x + b * y + c) / d)
        
        # Update parameters
        a -= learning_rate * da
        b -= learning_rate * db
        c -= learning_rate * dc

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
a, b, c = gradient_descent(x, y)

print(f"Fitted line parameters: a = {a}, b = {b}, c = {c}")

# Plotting the data points and the fitted line
plt.scatter(x, y, color='blue', label='Data points')

# Calculate fitted line points
x_vals = np.linspace(min(x), max(x), 100)
y_vals = -(a * x_vals + c) / b

plt.plot(x_vals, y_vals, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Line Fitting using Newton\'s Method')
plt.show()