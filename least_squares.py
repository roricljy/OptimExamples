import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

def fit_quadratic_surface(image):
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

def main(image_path):
    # Load the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not open or find the image!")
        return

    # Fit and remove the background
    background = fit_quadratic_surface(image)
    residual = background.astype(np.float32) - image.astype(np.float32)
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    _, binarized = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Display results
    plt.figure(figsize=(6, 6))
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