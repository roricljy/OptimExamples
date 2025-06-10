import numpy as np
import math
import time
from tkinter import Tk, Canvas, Button
import os
gscale = 3 if "ANDROID_STORAGE" in os.environ else 1

# Global variables to store points and canvas
points = []
canvas = None
optim_started = False
stop_iteration = False

def click_event(event):
    global points, canvas
    x, y = event.x, event.y
    points.append((x, y))
    canvas.create_oval(x - 5*gscale, y - 5*gscale, x + 5*gscale, y + 5*gscale, fill="#00FF00")

def on_run_btn():
    global optim_started, stop_iteration
    if not optim_started:
        optim_started = True
        fit_sine_curve()
        root.destroy()
        return
    stop_iteration = True
    
def key_event(event):
    global optim_started, stop_iteration
    if not optim_started:
        optim_started = True
        fit_sine_curve()
        root.destroy()
        return
    stop_iteration = True
        
def optim_sinefit(option, points, learning_rate=0.1, damping=1.0, tolerance=1e-20, max_iter=10000):
    global canvas, stop_iteration
    stop_iteration = False
    points_array = np.array(points)
    x, y = points_array[:, 0], points_array[:, 1]

    # Initial guess: y = a * sin(b * x + c) + d
    a = np.std(y)
    b = estimate_hz(x, y)
    c = 0
    d = np.mean(y)
    params = np.array([a, b, c, d])

    msg = []
    for iter in range(max_iter):
        a, b, c, d = params
        res = a * np.sin(b * x + c) + d - y
        J = np.vstack([
            np.sin(b * x + c),
            a * np.cos(b * x + c) * x,
            a * np.cos(b * x + c),
            np.ones(len(x))
        ]).T
        if option == 1:
            delta = np.linalg.lstsq(J, res, rcond=None)[0]  # Gauss-Newton
            msg = "Gauss-Newton"
        elif option == 2:
            delta = np.linalg.pinv(J.T @ J + np.eye(len(params)) * damping) @ J.T @ res  # Levenberg
            msg = "Levenberg"
        else:
            delta = np.linalg.pinv(J.T @ J + np.diag(np.diag(J.T @ J)) * damping) @ J.T @ res  # LM
            msg = "Levenberg-Marquardt"
        params -= learning_rate * delta
        if np.linalg.norm(delta) < tolerance:
            break

        loss = np.linalg.norm(res, ord=2)
        print(f"[{iter}] loss = {loss}")
        plot_sine(params, points, msg)
        
        if stop_iteration:
            return params       
        
    return params

def plot_sine(params, points, msg):
    a, b, c, d = params
    x = np.linspace(0, 853*gscale, 853*gscale)
    y = a * np.sin(b * x + c) + d
    canvas.delete("all")
    for i in range(1, len(x)):
        canvas.create_line(x[i-1], y[i-1], x[i], y[i], fill="white", width=2*gscale)
    for j in range(0, len(points)):
        x1, y1 = points[j]
        canvas.create_oval(x1 - 5*gscale, y1 - 5*gscale, x1 + 5*gscale, y1 + 5*gscale, fill="#00FF00")
    canvas.create_text(30*gscale, 25*gscale, text=msg, fill="white", font=("Helvetica", 24), anchor="nw")
    canvas.update()
    time.sleep(0.001)

def fit_sine_curve():
    global points
    params_est1 = optim_sinefit(1, points, learning_rate=0.1, damping=1.0)
    print(f"Fitted Parameters (Gauss-Newton): {params_est1}")
    params_est2 = optim_sinefit(2, points, learning_rate=0.1, damping=1.0)    
    print(f"Fitted Parameters (Levenberg): {params_est2}")
    params_est3 = optim_sinefit(3, points, learning_rate=0.1, damping=1.0)    
    print(f"Fitted Parameters (LM): {params_est3}")

def estimate_hz(x, y):
    my = np.mean(y)
    n_cross = 0
    for i in range(1, len(y)):
        if (y[i-1]>my and y[i]<=my) or (y[i-1]<my and y[i]>=my):
            n_cross = n_cross + 1
    x_interval = np.max(x) - np.min(x)
    period = x_interval / (n_cross*0.5)
    return 2*math.pi / period

# Tkinter GUI setup
root = Tk()
root.title("Sine Curve Fitting")

canvas = Canvas(root, width=853*gscale, height=480*gscale, bg="black")
canvas.pack()
canvas.bind("<Button-1>", click_event)
root.bind("<Key>", key_event)

points = [(76, 36), (74, 349), (146, 445), (276, 424), (294, 140), (360, 30), (512, 81), (535, 254), (557, 353), (642, 382), (722, 300), (756, 70)]
points = points * gscale
for pt in points:
    canvas.create_oval(pt[0] - 5*gscale, pt[1] - 5*gscale, pt[0] + 5*gscale, pt[1] + 5*gscale, fill="#00FF00")

# Add button
run_btn = Button(root, text="Run", command=on_run_btn)
run_btn.pack()
root.mainloop()
