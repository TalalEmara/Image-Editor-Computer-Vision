import cv2
import numpy as np

def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel.

    Parameters:
    - size: The size of the kernel (must be an odd number).
    - sigma: The standard deviation of the Gaussian distribution.

    Returns:
    - A 2D numpy array representing the Gaussian kernel.
    """
    k = (size - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def convolve2d(image, kernel):
    """
    Apply a 2D convolution operation between an image and a kernel.

    Parameters:
    - image: The input image as a 2D numpy array.
    - kernel: The convolution kernel as a 2D numpy array.

    Returns:
    - The convolved image as a 2D numpy array.
    """
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output array
    output = np.zeros_like(image)

    # Perform the convolution
    for i in range(i_height):
        for j in range(i_width):
            region = padded_image[i:i + k_height, j:j + k_width]
            output[i, j] = np.sum(region * kernel)

    return output

import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel.

    Parameters:
    - size: The size of the kernel (must be an odd number).
    - sigma: The standard deviation of the Gaussian distribution.

    Returns:
    - A 2D numpy array representing the Gaussian kernel.
    """
    k = (size - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def convolve2d(image, kernel):
    """
    Apply a 2D convolution operation between an image and a kernel.

    Parameters:
    - image: The input image as a 2D numpy array.
    - kernel: The convolution kernel as a 2D numpy array.

    Returns:
    - The convolved image as a 2D numpy array.
    """
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output array
    output = np.zeros_like(image)

    # Perform the convolution
    for i in range(i_height):
        for j in range(i_width):
            region = padded_image[i:i + k_height, j:j + k_width]
            output[i, j] = np.sum(region * kernel)

    return output

def hybrid_image(image1, image2,sigma=8):
    """
    Create a hybrid image by combining the low-pass filtered version of image1
    with the high-pass filtered version of image2.

    Parameters:
    - image1: First input image (2D numpy array).
    - image2: Second input image (2D numpy array).

    Returns:
    - Hybrid image as a 2D numpy array.
    """
    # Ensure both images have the same dimensions
    assert image1.shape == image2.shape, "Input images must have the same dimensions."

    # Determine default sigma based on image dimensions
    # sigma = 8  # Default sigma as 1/16th of the smaller image dimension

    # Calculate kernel size based on sigma
    kernel_size = 30  # Ensures the kernel captures >99% of the Gaussian distribution

    # Generate the Gaussian kernel
    gaussian_k = gaussian_kernel(kernel_size, sigma)

    # Apply Gaussian filter to image1 to extract low-frequency components
    low_frequencies = convolve2d(image1, gaussian_k)

    # Apply Gaussian filter to image2
    blurred_image2 = convolve2d(image2, gaussian_k)

    # Extract high-frequency components from image2
    high_frequencies = image2 - blurred_image2

    # Combine low and high frequencies
    hybrid = low_frequencies + high_frequencies

    # Clip values to ensure they remain within valid range
    hybrid = np.clip(hybrid, 0, 255)

    # Display the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(low_frequencies, cmap='gray')
    axes[0].set_title('Low Frequencies (Image 1)')
    axes[0].axis('off')

    axes[1].imshow(high_frequencies + 127.5, cmap='gray')  # Shift to visualize both positive and negative values
    axes[1].set_title('High Frequencies (Image 2)')
    axes[1].axis('off')

    axes[2].imshow(hybrid, cmap='gray')
    axes[2].set_title('Hybrid Image')
    axes[2].axis('off')

    # plt.show()

    return hybrid.astype(np.uint8)

# # Example usage:
# # Load images using OpenCV
# image1 = cv2.imread('C:/Faculty/SBE 24-25/Computer Vision/Repo/Image-Editor-Computer-Vision/images/dog.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('C:/Faculty/SBE 24-25/Computer Vision/Repo/Image-Editor-Computer-Vision/images/cat.jpg', cv2.IMREAD_GRAYSCALE)

# # Ensure images are loaded successfully
# if image1 is None or image2 is None:
#     raise ValueError("One or both images could not be loaded. Check the file paths.")

# # Create hybrid image
# hybrid = hybrid_image(image1, image2)


