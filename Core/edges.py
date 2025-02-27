import cv2
import numpy as np
import matplotlib.pyplot as plt
from gray import rgb_to_grayscale
from filters import gaussian_filter

# convolution
def custom_convolution(image, kernel):
    # Filter image
    image = gaussian_filter(image, 5, 10)
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

def generate_sobel_kernels(kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Create a grid of indices centered around the middle of the kernel
    radius = kernel_size // 2
    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]

    # Horizontal kernel (Gx)
    Gx = x * (1 / (2 * np.pi * (x**2 + y**2 + 1e-6)))  # Avoid division by zero
    Gx[:, radius] = 0  # Central column is zero

    # Vertical kernel (Gy)
    Gy = y * (1 / (2 * np.pi * (x**2 + y**2 + 1e-6)))  # Avoid division by zero
    Gy[radius, :] = 0  # Central row is zero

    # Normalize the kernels
    Gx /= np.sum(np.abs(Gx))
    Gy /= np.sum(np.abs(Gy))

    return Gx, Gy

def generate_roberts_kernels():
    Gx = np.array([[1, 0], [0, -1]])
    Gy = np.array([[0, 1], [-1, 0]])
    Gx = Gx / np.sum(np.abs(Gx))
    Gy = Gy / np.sum(np.abs(Gy))
    return Gx, Gy

def generate_prewitt_kernels(kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Horizontal kernel (Gx)
    Gx = np.ones((kernel_size, kernel_size))
    Gx[:, kernel_size//2] = 0  # Central column is zero
    for i in range((kernel_size//2)+1, kernel_size):
        Gx[:, i] *= -1
    Gx = Gx / np.sum(np.abs(Gx))

    # Vertical kernel (Gy)
    Gy = np.ones((kernel_size, kernel_size))
    Gy[kernel_size//2, :] = 0  # Central row is zero
    for i in range((kernel_size//2)+1, kernel_size):
        Gy[i, :] *= -1
    Gy = Gy / np.sum(np.abs(Gy))

    return Gx, Gy

def sobel(image, kernel_size=3):
    Gx, Gy = generate_sobel_kernels(kernel_size)
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

def robert(image):
    Gx, Gy = generate_roberts_kernels()
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

def prewitt(image, kernel_size=3):
    Gx, Gy = generate_prewitt_kernels(kernel_size)
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

# Canny 
def canny(image, low_threshold=50, high_threshold=100):
    # filter image
    image = gaussian_filter(image, 5, 1)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

# using OpenCV
def apply_edge_filters(image):
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel Filter
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Roberts Filter
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    roberts_gx = cv2.filter2D(image, cv2.CV_64F, roberts_x)
    roberts_gy = cv2.filter2D(image, cv2.CV_64F, roberts_y)
    roberts_edges = np.sqrt(roberts_gx**2 + roberts_gy**2)
    roberts_edges = cv2.normalize(roberts_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Prewitt Filter
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    prewitt_gx = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
    prewitt_gy = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
    prewitt_edges = np.sqrt(prewitt_gx**2 + prewitt_gy**2)
    prewitt_edges = cv2.normalize(prewitt_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return sobel_edges, roberts_edges, prewitt_edges

# --------------- TESTING ---------------
image_path = "CV/Image-Editor-Computer-Vision/images/colored2.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
imageRGB = cv2.imread(image_path)
imageRGB = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2RGB)

# Apply Sobel filter using OpenCV
_, _, sobelimg, _ = sobel(imageRGB, 5)
_, _, robertimg, _ = robert(imageRGB)
_, _, prewittimg, _ = prewitt(imageRGB, 5)
cannyimg = canny(imageRGB)

sobelimg2, robertimg2, prewittimg2 = apply_edge_filters(imageRGB)

plt.figure(figsize=(12, 6))

# First Row: Original Image and Filters (First Set)
plt.subplot(2, 5, 1)
plt.title("Original Image")
plt.imshow(imageRGB)
plt.axis('off')

plt.subplot(2, 5, 2)
plt.title("Sobel")
plt.imshow(sobelimg, cmap='gray')
plt.axis('off')

plt.subplot(2, 5, 3)
plt.title("Robert")
plt.imshow(robertimg, cmap='gray')
plt.axis('off')

plt.subplot(2, 5, 4)
plt.title("Prewitt")
plt.imshow(prewittimg, cmap='gray')
plt.axis('off')

plt.subplot(2, 5, 5)
plt.title("Canny")
plt.imshow(cannyimg, cmap='gray')
plt.axis('off')

# Second Row: Filters (Second Set)
plt.subplot(2, 5, 7)
plt.title("Sobel 2")
plt.imshow(sobelimg2, cmap='gray')
plt.axis('off')

plt.subplot(2, 5, 8)
plt.title("Robert 2")
plt.imshow(robertimg2, cmap='gray')
plt.axis('off')

plt.subplot(2, 5, 9)
plt.title("Prewitt 2")
plt.imshow(prewittimg2, cmap='gray')
plt.axis('off')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()