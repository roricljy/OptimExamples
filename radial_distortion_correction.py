import cv2
import numpy as np
import math

# Global variables to store points and image
points = []
image = None

# Mouse callback function to capture points
def click_event(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) <= 3:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", image)

# Function to apply the radial distortion
def apply_radial_distortion(point, center, w):
    if w == 0 or point == center:
        return point
    ru = np.linalg.norm(np.array(point) - np.array(center))
    rd = np.arctan(2 * ru * np.tan(w / 2)) / w
    distorted = (np.array(point) - np.array(center)) * rd / ru + np.array(center)
    return distorted

# Function to correct the radial distortion
def correct_radial_distortion(point, center, w):
    if w == 0 or point == center:
        return point
    rd = np.linalg.norm(np.array(point) - np.array(center))
    ru = np.tan(rd * w) / (2 * np.tan(w / 2))
    corrected = (np.array(point) - np.array(center)) * ru / rd + np.array(center)
    return corrected

# Objective function to minimize
def objective_function(w, points, center):
    p1 = correct_radial_distortion(points[0], center, w)
    p2 = correct_radial_distortion(points[1], center, w)
    p3 = correct_radial_distortion(points[2], center, w)

    v1 = (np.array(p2) - np.array(p1)) / np.linalg.norm(np.array(p2) - np.array(p1))
    v2 = (np.array(p3) - np.array(p1)) / np.linalg.norm(np.array(p3) - np.array(p1))
    loss = np.linalg.norm(np.cross(v1, v2))
    return loss

# Gradient Descent implementation
def gradient_descent(points, center, learning_rate=1e-6, tolerance=1e-12, max_iter=1000):
    w = 0
    eps = 1e-7
    for iter in range(max_iter):
        w_eps = w + eps
        grad = (objective_function(w_eps, points, center) - objective_function(w, points, center)) / eps
        new_w = w - learning_rate * grad

        delta = abs(new_w - w)
        if delta < tolerance:
            break

        w = new_w
        loss = objective_function(w, points, center)
        print(f"[{iter}] loss = {loss}, w = {w}")
    return w

# Main function
if __name__ == "__main__":
    # Read the image
    image = cv2.imread("sample_radial.png")
    if image is None:
        print("Image not found!")
        exit(-1)

    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Image", click_event)

    print("Click on three points in the image")
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    if len(points) < 3:
        print("Not enough points selected!")
        exit(-1)

    # Perform gradient descent to find k1 and k2
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    w = gradient_descent(points, center)
    print(f"Found radial distortion coefficients: w = {w}")

    # Apply distortion correction to the image
    new_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            distorted_point = apply_radial_distortion((x, y), center, w)
            xd = int(round(distorted_point[0]))
            yd = int(round(distorted_point[1]))
            if 0 <= xd < width and 0 <= yd < height:
                new_image[y, x] = image[yd, xd]

    cv2.imshow("Corrected Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()