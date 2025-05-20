import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Function to compute gradient and Hessian of the objective function
# J = sum{(ax + by + c)^2/(a^2 + b^2)}
def compute_gradient_hessian(x, y, a, b, c):
    # Compute the common numerator L and denominator d
    L = a*x + b*y + c
    d = a**2 + b**2

    # Compute the gradients
    da = 2 * np.sum(L * (x * d - a * L) / d**2)
    db = 2 * np.sum(L * (y * d - b * L) / d**2)
    dc = 2 * np.sum(L / d)
    G = np.array([da, db, dc])

    # Compute the Hessian    
    H = np.zeros((3, 3))
    H[0, 0] = 2 * np.sum((x**2)/d - (L**2)/d**2 + 4*(a**2)*(L**2)/d**3 - (4*a*x*L)/d**2)
    H[0, 1] = H[1, 0] = 2 * np.sum((x*y)/d - (2*a*y*L)/d**2 - (2*b*x*L)/d**2 + 4*a*b*(L**2)/d**3)
    H[0, 2] = H[2, 0] = 2 * np.sum(-(x*a**2 + 2*y*a*b + 2*c*a - x*b**2)/d**2)
    H[1, 1] = 2 * np.sum((y**2)/d - (L**2)/d**2 + (4*b**2)*(L**2)/d**3 - (4*b*y*L)/d**2)
    H[1, 2] = H[2, 1] = 2 * np.sum(-(-y*a**2 + 2*x*a*b + y*b**2 + 2*c*b)/d**2)
    H[2, 2] = 2 * np.sum(1/d)

    return G, H

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
def newton_method(x, y, learning_rate=0.9, num_iterations=300):
    # Initial parameters a, b, c
    a = 1.0
    b = 1.0
    c = 1.0

    font_bkg=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0")
    for iter in range(num_iterations):
        plt.clf()

        # Update parameters using Newton's method
        gradients, H = compute_gradient_hessian(x, y, a, b, c)
        eigvals, eigvecs = np.linalg.eigh(H)
        H_abs = eigvecs @ np.diag(np.abs(eigvals)) @ eigvecs.T
        delta = np.linalg.solve(H, gradients)                      # Newton's method
        #delta = np.linalg.solve(H + np.eye(3) * 0.1, gradients)  # Newton + Damping
        #delta = np.linalg.solve(H_abs, gradients)                 # Saddle-free Newton
        #delta = np.linalg.solve(H_abs + np.eye(3) * 0.1, gradients)  # Saddle-free + Damping               

        # Update parameters using Gauss-Newton method
        res = np.abs(x * a + y * b + c) / np.sqrt(a**2 + b**2)
        J = compute_Jacobian(x, y, a, b, c)
        #delta = np.linalg.lstsq(J, res, rcond=None)[0] # Gauss-Newton

        a -= learning_rate * delta[0]
        b -= learning_rate * delta[1]
        c -= learning_rate * delta[2]

        # Print loss
        loss = np.sum((a*x + b*y + c)**2/(a**2 + b**2))        
        plt.text(0.3, 0.3, f"Iter={iter}: loss = {loss:0.3f}", transform=plt.gca().transAxes, fontsize=18*gscale, color="black", bbox=font_bkg)            

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
        plt.title('Line Fitting: Newton\'s method')
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