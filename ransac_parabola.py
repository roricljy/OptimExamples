import numpy as np
import matplotlib.pyplot as plt

# Data generation
outlier_ratio = 0.7
sigma = 100
x0 = np.arange(1, 101)
y0 = -2 * (x0 - 40)**2 + 30 + sigma * np.random.randn(len(x0))
n_outlier = round(len(x0) * outlier_ratio / (1 - outlier_ratio))
x = np.concatenate([x0, np.random.rand(n_outlier) * max(x0)])
y = np.concatenate([y0, np.random.rand(n_outlier) * (max(y0) - min(y0)) * 1.2 + 1.1 * min(y0)])
A0 = np.vstack([x0**2, x0, np.ones(len(x0))]).T
A = np.vstack([x**2, x, np.ones(len(x))]).T

# RANSAC fitting
n_data = len(x)
N = 1000      # iterations
T = 2 * sigma   # residual threshold
n_sample = 3
max_cnt = 0
best_model = np.zeros(3)

for itr in range(N):
    k = np.random.randint(0, n_data, n_sample)
    Ak = A[k]
    pk = np.linalg.pinv(Ak).dot(y[k])
    residual = np.abs(A.dot(pk) - y)
    cnt = len(residual[residual < T])
    if cnt > max_cnt:
        best_model = pk
        max_cnt = cnt

# Optional LS (Least Squares) fitting
residual = np.abs(y - A.dot(best_model))
in_k = np.where(residual < T)[0]
A2 = A[in_k]
p = np.linalg.pinv(A2).dot(y[in_k])  # final model

# Drawing
plt.plot(x0, A0.dot(p), 'r', linewidth=4, label='Fitted curve')
plt.plot(x, y, '*b', label='Data points')
plt.legend()
plt.show()