import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def fit_parabola(points):
    n = len(points)
    A = np.zeros((n, 3))
    B = np.zeros(n)
    for i, (x, y) in enumerate(points):
        A[i] = [x**2, x, 1]
        B[i] = y
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return coeffs

def point_to_parabola_distance(point, coeffs):
    x, y = point
    y_fit = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
    return abs(y - y_fit)

def ransac_parabola(points, max_iter, threshold, min_inliers):
    best_inlier_count = 0
    best_model = None

    for _ in range(max_iter):
        sample = random.sample(points, 3)
        model = fit_parabola(sample)

        inlier_count = 0
        for point in points:
            if point_to_parabola_distance(point, model) < threshold:
                inlier_count += 1

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = model

    inliers = [point for point in points if point_to_parabola_distance(point, best_model) < threshold]
    if len(inliers) >= min_inliers:
        best_model = fit_parabola(inliers)

    return best_model

def main():
    # Generate some sample data with outliers
    points = [
        (0, 1), (1, 2), (2, 5), (3, 10), (4, 17), (5, 26),
        (1, 8), (2, -1), (3, 4), (5, 12), (6, 30), (7, 5)
    ]

    # RANSAC parameters
    max_iter = 1000
    threshold = 1.0
    min_inliers = 5

    # Fit the parabola using RANSAC
    model = ransac_parabola(points, max_iter, threshold, min_inliers)

    # Print the model coefficients
    print(f"Fitted parabola: y = {model[0]}x^2 + {model[1]}x + {model[2]}")

    # Visualize the points and the fitted parabola
    plot = np.full((400, 600, 3), 255, dtype=np.uint8)
    for x, y in points:
        cv2.circle(plot, (int(x * 50 + 50), int(400 - y * 10 - 50)), 5, (0, 0, 255), -1)
    for x in np.linspace(0, 10, 100):
        y = model[0] * x**2 + model[1] * x + model[2]
        cv2.circle(plot, (int(x * 50 + 50), int(400 - y * 10 - 50)), 3, (255, 0, 0), -1)

    cv2.imshow("RANSAC Parabola Fitting", plot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()