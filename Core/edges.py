"""
Use sobel, robert, prewitt, canny functions only
The other functions are called in these functions
All errors are handled in these function ان شاء الله
Testing and builtin functions are done at the end of the file
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from Core.gray import rgb_to_grayscale
from Core.filters import gaussian_filter

# convolution
def custom_convolution(image, kernel):
    # Filter image
    if len(image.shape) == 3:
        image = rgb_to_grayscale(image)
    image = gaussian_filter(image, 3, 1.5)
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

    if kernel_size == 3:
        # Standard 3x3 Sobel kernels
        Gx = np.array([[1, 0, -1], 
                       [2, 0, -2], 
                       [1, 0, -1]])
        
        Gy = np.array([[1, 2, 1], 
                       [0, 0, 0], 
                       [-1, -2, -1]])
    else:
        # For larger kernels, use a different approach
        # This is a simplified example - OpenCV uses more complex methods
        Gx = np.zeros((kernel_size, kernel_size))
        Gy = np.zeros((kernel_size, kernel_size))
        
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if j < center:
                    Gx[i, j] = -(center - j + 1)
                elif j > center:
                    Gx[i, j] = (j - center + 1)
                
                if i < center:
                    Gy[i, j] = -(center - i + 1)
                elif i > center:
                    Gy[i, j] = (i - center + 1)
    Gx = Gx / np.sum(np.abs(Gx))
    Gy = Gy / np.sum(np.abs(Gy))
    return Gx, Gy

def generate_roberts_kernels():
    Gx = np.array([[1, 0], [0, -1]])
    Gy = np.array([[0, 1], [-1, 0]])
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
    """
    image: rgb image
    kernel_size: size of the kernel (odd number)
    returns gradient_x, gradient_y, gradient_magnitude, gradient_direction
    (use gradient_magnitude for edge detection)
    """
    Gx, Gy = generate_sobel_kernels(kernel_size)
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_magnitude = custom_normalize(gradient_magnitude, 0, 255)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

def robert(image):
    """
    image: rgb image
    ""no kernel size needed""
    returns gradient_x, gradient_y, gradient_magnitude, gradient_direction
    (use gradient_magnitude for edge detection)
    """
    Gx, Gy = generate_roberts_kernels()
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_magnitude = custom_normalize(gradient_magnitude, 0, 255)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

def prewitt(image, kernel_size=3):
    """
    image: rgb image
    kernel_size: size of the kernel (odd number)
    returns gradient_x, gradient_y, gradient_magnitude, gradient_direction
    (use gradient_magnitude for edge detection)
    """
    Gx, Gy = generate_prewitt_kernels(kernel_size)
    gradient_x = custom_convolution(image, Gx)
    gradient_y = custom_convolution(image, Gy)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_magnitude = custom_normalize(gradient_magnitude, 0, 255)
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

# Canny 
def canny(image, low_threshold=50, high_threshold=100):
    """
    image: rgb image 
    low_threshold: lower threshold for edge detection
    high_threshold: higher threshold for edge detection
    returns 2d edge detected image
    """
    # filter image
    image = gaussian_filter(image, 5, 1)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges.astype(np.uint8)

def custom_normalize(image, out_min=0, out_max=255):
    """
    Normalize image values to a specific range without using OpenCV.
    """
    # Handle empty images
    if image is None or image.size == 0:
        return None
        
    # Get min and max values
    img_min = np.min(image)
    img_max = np.max(image)
    
    # Avoid division by zero
    if img_max == img_min:
        return np.ones_like(image) * out_min
    
    # Normalize to specified range
    normalized = (image - img_min) * (out_max - out_min) / (img_max - img_min) + out_min
    return normalized.astype(np.uint8)

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
    # prewitt_edges = cv2.normalize(prewitt_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return sobel_edges, roberts_edges, prewitt_edges

# --------------- TESTING ---------------
def test_edge_filters():
    image_path = "CV/Image-Editor-Computer-Vision/images/bobama.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    imageRGB = cv2.imread(image_path)
    imageRGB = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2RGB)

    # Apply Sobel filter using OpenCV
    _, _, sobelimg, _ = sobel(imageRGB, 3)
    _, _, robertimg, _ = robert(imageRGB)
    _, _, prewittimg, _ = prewitt(imageRGB, 3)
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

# test_edge_filters()