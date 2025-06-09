import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

x = np.arange(1, 101)
y = -2 * (x - 40)**2 + 30 + 100 * np.random.randn(100)
y[50:80] += 400 * np.abs(np.arange(50, 80) - 65) - 7000
n = len(x)

a = cp.Variable()
b = cp.Variable()
c = cp.Variable()

y_pred = a * x**2 + b * x + c

objective = cp.Minimize(cp.norm1(y - y_pred))

prob = cp.Problem(objective)
prob.solve()

print(f"L1 : y = {a.value:.4f}x^2 + {b.value:.4f}x + {c.value:.4f}")

x_fit = np.linspace(min(x), max(x), 200)
y_fit = a.value * x_fit**2 + b.value * x_fit + c.value

plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_fit, y_fit, color='green', label='L1 Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('L1 Norm Quadratic Fit')
plt.legend()
plt.grid(False)
plt.show()