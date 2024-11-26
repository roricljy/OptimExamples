import numpy as np
import math
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, filedialog
plt.rcParams.update({'font.size': 14})

# Global variables
points = []
image = None
canvas_img = None
canvas_widget = None

# Load an image using a file dialog
def load_image():
    global image, canvas_img, canvas_widget
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = plt.imread(file_path)
    if img is not None:
        image = img
        canvas_img = np.copy(image)
        display_image_on_canvas()

# Display the image on the Tkinter canvas
def display_image_on_canvas():
    global canvas_widget, canvas_img
    if canvas_widget is not None:
        canvas_widget.delete("all")  # Clear existing canvas content
    height, width = canvas_img.shape[:2]
    # Resize the canvas to match the image size
    canvas.config(width=width, height=height)
    # Convert image array to PhotoImage-compatible format
    from PIL import Image, ImageTk
    img_pil = Image.fromarray((canvas_img * 255).astype(np.uint8))  # Ensure values are 0-255
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas_widget = canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_tk = img_tk  # Keep a reference to prevent garbage collection

# Mouse callback function to capture points
def click_event(event):
    global points
    if len(points) < 3:
        x, y = event.x, event.y
        points.append((x, y))
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")
        if len(points) == 3:
            process_points()

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
    
# Perform gradient descent to find optimal w
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

# Process the selected points and show corrected image
def process_points():
    global points, image
    if len(points) < 3:
        print("Not enough points selected!")
        return

    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Perform gradient descent
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

    # Show corrected image
    plt.imshow(new_image)
    plt.title("Corrected Image")
    plt.axis('off')
    plt.show()

# Initialize Tkinter
root = Tk()
root.title("Radial Distortion Correction")

# Create canvas for image display
canvas = Canvas(root)
canvas.pack()

# Bind mouse click events to the canvas
canvas.bind("<Button-1>", click_event)

# Add buttons
load_btn = Button(root, text="Load Image", command=load_image)
load_btn.pack()

root.mainloop()