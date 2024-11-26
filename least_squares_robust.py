import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
gscale = 2 if "ANDROID_STORAGE" in os.environ else 1
plt.rcParams.update({'font.size': 14*gscale})

def cauchy_weight(residual, c):
    return 1.0 / (1.0 + (residual / c)**2)

def fit_quadratic_surface_robust(image, c, iterations=20):
    rows, cols = image.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z = image

    # Prepare the data for least squares fitting
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    A = np.vstack([X_flat**2, Y_flat**2, X_flat*Y_flat, X_flat, Y_flat, np.ones_like(X_flat)]).T
    B = Z_flat    

    # Initial guess for the coefficients (least squares)
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Iterative reweighted least squares
    for itr in range(iterations):
        residuals = A @ coeffs - B
        
        weights = cauchy_weight(residuals, c)
        W = np.diag(weights)

        Aw = A.T @ W @ A
        Bw = A.T @ W @ Z_flat
        coeffs = np.linalg.solve(Aw, Bw)

        # Create the background model
        background = (coeffs[0] * X**2 + coeffs[1] * Y**2 +
                      coeffs[2] * X * Y + coeffs[3] * X +
                      coeffs[4] * Y + coeffs[5])
                      
        residual = background.astype(np.float32) - image.astype(np.float32)
        residual = np.clip(residual, 0, 255).astype(np.uint8)
        _, binarized = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)                      
                      
        # Plot the intermediate result
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.title(f'Iteration: #{itr + 1}')
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.title('Background')
        plt.imshow(background, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.title('Residual Image')
        plt.imshow(residual, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.title('Binary Image')
        plt.imshow(binarized, cmap='gray')
        plt.axis('off')
        plt.pause(0.1)

    return background

def main(image_path):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not open or find the image!")
        return

    # Initialize display
    plt.figure(figsize=(7*gscale, 7*gscale))
    plt.pause(0.1)

    # Fit and remove the background
    c = 2.3849  # Cauchy function parameter
    background = fit_quadratic_surface_robust(image, c)
    residual = background.astype(np.float32) - image.astype(np.float32)
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    _, binarized = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Display final results
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title('Background')
    plt.imshow(background, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title('Residual Image')
    plt.imshow(residual, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title('Binary Image')
    plt.imshow(binarized, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main('sample_circle.png')