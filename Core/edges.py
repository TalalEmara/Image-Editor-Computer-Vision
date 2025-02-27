import cv2
import numpy as np
import matplotlib.pyplot as plt
from gray import rgb_to_grayscale
from filters import replicate_padding

# Sobal
def sobal(image, kernal_size=3, pad_size=1):
    # Convert the image to grayscale
    imageGray = rgb_to_grayscale(image)
    # Add padding
    paddedImage = replicate_padding(imageGray, pad_size)

    # Apply Sobel filters
    sobel_x = cv2.Sobel(paddedImage, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobel_y = cv2.Sobel(paddedImage, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

    # Compute the gradient magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))  # Normalize

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.imshow(sobel_x, cmap="gray"), plt.title("Sobel X")
    plt.subplot(1, 3, 2), plt.imshow(sobel_y, cmap="gray"), plt.title("Sobel Y")
    plt.subplot(1, 3, 3), plt.imshow(sobel_magnitude, cmap="gray"), plt.title("Sobel Magnitude")
    plt.show()

# Robert
def robert(image):
    pass    

# Prewitt
def prewitt(image):
    pass

# Canny 
def canny(image):
    pass


# --------------- TESTING ---------------
image_path = "CV/Image-Editor-Computer-Vision/images/colored2.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
imageRGB = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

sobal(imageRGB)