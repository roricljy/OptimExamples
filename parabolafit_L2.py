import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 101)
y = -2 * (x - 40)**2 + 30 + 100 * np.random.randn(100)
#y[50:80] += 400 * np.abs(np.arange(50, 80) - 65) - 7000
n = len(x)

A = np.vstack([x**2, x, np.ones_like(x)]).T

coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
a, b, c = coefficients

print(f"L2 : y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}")

x_fit = np.linspace(min(x), max(x), 200)
y_fit = a * x_fit**2 + b * x_fit + c

plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_fit, y_fit, color='green', label='L2 Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('L2 Norm Quadratic Fit')
plt.legend()
plt.grid(False)
plt.show()