import cv2
import numpy as np
import matplotlib.pyplot as plt
from gray import rgb_to_grayscale
from filters import gaussian_filter

# convolution
def custom_convolution(image, kernel):
    # Filter image
    image = rgb_to_grayscale(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Pad the image to handle edges
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='symmetric')
    # Initialize output array
    output = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest from the padded image
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)

    return output

# Sobal
def sobel(image, kernel_size=3):
    # Generate Sobel kernels
    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = Gx / np.sum(np.abs(Gx))
    Gy = Gy / np.sum(np.abs(Gy))
    # Apply custom convolution to compute gradients
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)

    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

# Roberts
def robert(image):
    # if image.ndim != 2:
    #     image = rgb_to_grayscale(image)

    # Define Roberts kernels
    Gx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    Gy = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Compute gradients using custom convolution
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)

    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_x, gradient_y, gradient_magnitude, gradient_direction   

# Prewitt
def prewitt(image):
    # Define Roberts kernels
    Gx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)


    # Compute gradients using custom convolution
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)

    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_x, gradient_y, gradient_magnitude, gradient_direction   


# Canny 
def canny(image, low_threshold=50, high_threshold=150):
    if len(image.shape) != 2:
        image = rgb_to_grayscale(image)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


# --------------- TESTING ---------------
image_path = "CV/Image-Editor-Computer-Vision/images/colored2.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
imageRGB = cv2.imread(image_path)
imageRGB = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2RGB)

# Display Results

# Apply Sobel filter using OpenCV
gradient_x, gradient_y, gradient_magnitude, gradient_direction = sobel(imageRGB)
# edge = canny(imageRGB)
# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(imageRGB)

plt.subplot(1, 4, 2)
plt.title("Gradient X")
plt.imshow(gradient_x, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Gradient Y")
plt.imshow(gradient_y, cmap='gray')

plt.subplot(1, 4, 4)
plt.title("Gradient Magnitude")
plt.imshow(gradient_magnitude, cmap='gray')

plt.show()