import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Gradient descent function to fit a line ax + by + c = 0
def gradient_descent(x, y, learning_rate=0.01, num_iterations=300):
    # Initial parameters a, b, c
    a = 1.0
    b = 1.0
    c = 1.0

    for iter in range(num_iterations):
        plt.clf()
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
        bkg=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
        plt.text(0.4, 0.1, f"Iter={iter}: loss = {loss:0.3f}", transform=plt.gca().transAxes, fontsize=18*gscale, color="black", bbox=bkg)    

        # Plotting the data points and the fitted line        
        plt.scatter(x, y, color='blue', s=100*gscale, label='Data points')

        # Calculate fitted line points
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = -(a * x_vals + c) / b

        plt.plot(x_vals, y_vals, color='red', linewidth=3*gscale, label='Fitted line')
        plt.xlim(0, 6)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Line Fitting: Gradient Descent')
        plt.pause(0.1)
   
    return a, b, c

# Example usage
# Sample data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2, 3, 5, 4])

# Init display
plt.figure(figsize=(7*gscale, 7*gscale))
plt.pause(0.1)

# Run gradient descent
a, b, c = gradient_descent(x, y)
plt.show()
