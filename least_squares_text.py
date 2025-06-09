import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas, Button, Frame, Label, Entry
from PIL import Image, ImageTk
import os
gscale = 1.8 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

# Global variables
image = None
image_show = None
canvas_widget = None

def fit_background(image):
    rows, cols = image.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z = image

    # Prepare the data for least squares fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    A = np.vstack([X_flat**2, Y_flat**2, X_flat*Y_flat, X_flat, Y_flat, np.ones_like(X_flat)]).T
    B = Z_flat

    # Solve for the coefficients
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Create the background model
    background = (coeffs[0] * X**2 + coeffs[1] * Y**2 +
                  coeffs[2] * X * Y + coeffs[3] * X +
                  coeffs[4] * Y + coeffs[5])

    return background

def display_image_on_canvas(canvas_img):
    global canvas_widget, gscale
    height, width = canvas_img.shape[:2]
    new_width, new_height = int(width * gscale), int(height * gscale)
    canvas.config(width=new_width, height=new_height)
    #canvas_img = (canvas_img * 255).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(canvas_img).resize((new_width, new_height), Image.NEAREST)
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image_tk = img_tk

def show_image():
    global image, image_show
    image_show = image
    display_image_on_canvas(image_show)

def show_bkg():
    global image, image_show
    image_bkg = fit_background(image)
    image_bkg = np.clip(image_bkg, 0, 255).astype(np.uint8)
    image_show = image_bkg
    display_image_on_canvas(image_show)

def show_residual():
    global image, image_show
    image_bkg = fit_background(image)
    residual = image_bkg.astype(np.float32) - image.astype(np.float32)
    residual -= residual.min()
    residual /= (residual.max()/255)
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    image_show = residual
    display_image_on_canvas(image_show)

def show_binary():
    global image_show
    if image_show is None:
        return
    _, binarized = cv2.threshold(image_show, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    display_image_on_canvas(binarized)

def exit_app():
    root.destroy()

if __name__ == "__main__":
    #image = plt.imread("sample_text.png")
    image = cv2.imread("sample_text.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found!")
        exit(-1)

    root = Tk()
    root.title("Text Segmentation")

    canvas = Canvas(root)
    canvas.pack()

    button_frame = Frame(root)
    button_frame.pack()

    load_btn = Button(button_frame, text="load img", command=show_image)
    load_btn.grid(row=0, column=0, padx=5, pady=5)
    
    fit_btn = Button(button_frame, text="fit bkg", command=show_bkg)
    fit_btn.grid(row=0, column=1, padx=5, pady=5)

    residual_btn = Button(button_frame, text="corrected(residual)", command=show_residual)
    residual_btn.grid(row=0, column=2, padx=5, pady=5)

    binary_btn = Button(button_frame, text="binarization", command=show_binary)
    binary_btn.grid(row=0, column=3, padx=5, pady=5)

    exit_btn = Button(button_frame, text="Exit", command=exit_app)
    exit_btn.grid(row=0, column=4, padx=5, pady=5)

    root.mainloop()
