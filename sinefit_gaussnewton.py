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
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)

# Gauss-Newton to fit sine function: y = a sin(bx + c) + d
# defults: learning_rate=0.1, dampling=10
def gauss_newton_sinefit(points, learning_rate=0.1, damping=10, tolerance=1e-12, max_iter=10000):
    # data points
    points_array = np.array(points)
    x, y = points_array[:, 0], points_array[:, 1]

    # Initial guess
    a = np.std(y)
    b = 0.02  #0.02
    c = 0      #0
    d = np.mean(y)
    params = np.array([a, b, c, d])

    for iter in range(max_iter):
        a, b, c, d = params
        res = a*np.sin(b*x + c) + d - y
        J = np.vstack([np.sin(b*x + c),
                       a*np.cos(b*x+c)*x,
                       a*np.cos(b*x+c),
                       np.ones(len(x))]).T        
        delta = np.linalg.lstsq(J, res, rcond=None)[0]
        #delta = np.linalg.pinv( J.T @ J + np.eye(len(params))*damping) @ J.T @ res
        #delta = np.linalg.pinv( J.T @ J + np.diag(np.diag( J.T @ J))*damping) @ J.T @ res
        params -= learning_rate * delta
        if np.linalg.norm(delta) < tolerance:
            break

        loss = np.linalg.norm(res, ord=2)
        print(f"[{iter}] loss = {loss}")
        
        draw = image.copy()
        plot_sine(draw, params)
        cv2.imshow('Image', draw)
        key = cv2.waitKey(1)
        if key != -1:
            return params
        
    return params

# Plot estimated sine function on the image
def plot_sine(draw, params):
    height, width, channels = draw.shape
    a, b, c, d = params
    x = np.linspace(0, width, width)
    y = a*np.sin(b*x + c) + d
    for i in range(1, len(x)):
        pt1 = (int(x[i - 1]), int(y[i - 1]))
        pt2 = (int(x[i]), int(y[i]))
        cv2.line(draw, pt1, pt2, (255, 255, 255), 2)
    
# Main function
if __name__ == "__main__":
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Image", click_event)

    print("Click on three points in the image")
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    if len(points) < 5:
        print("Not enough points selected!")
        exit(-1)

    # Sine fitting
    params_est = gauss_newton_sinefit(points)
    cv2.destroyAllWindows()