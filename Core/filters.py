import cv2
import numpy as np
import matplotlib.pyplot as plt
import gray
import NoiseAdder
def replicate_padding(image, pad_size):
    """Apply replicate padding to an image."""
    h, w = image.shape
    padded_image = np.zeros((h + 2 * pad_size, w + 2 * pad_size), dtype=image.dtype)

    # Copy the original image into the center
    padded_image[pad_size:pad_size + h, pad_size:pad_size + w] = image

    # Replicate edges
    padded_image[:pad_size, pad_size:-pad_size] = image[0, :]  # Top row
    padded_image[-pad_size:, pad_size:-pad_size] = image[-1, :]  # Bottom row
    padded_image[pad_size:-pad_size, :pad_size] = image[:, 0][:, np.newaxis]  # Left column
    padded_image[pad_size:-pad_size, -pad_size:] = image[:, -1][:, np.newaxis]  # Right column

    # Replicate corners
    padded_image[:pad_size, :pad_size] = image[0, 0]  # Top-left corner
    padded_image[:pad_size, -pad_size:] = image[0, -1]  # Top-right corner
    padded_image[-pad_size:, :pad_size] = image[-1, 0]  # Bottom-left corner
    padded_image[-pad_size:, -pad_size:] = image[-1, -1]  # Bottom-right corner

    return padded_image

def average_filter(image, kernel_size=5):
    """Apply an average filter."""
    pad_size = kernel_size // 2
    padded_image = replicate_padding(image, pad_size)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.mean(neighborhood)

    return filtered_image.astype(np.uint8)

def gaussian_kernel(size, sigma=1):
    """Generate a Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(ax, ax)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter(image, kernel_size=5, sigma=1.5):
    """Apply a Gaussian filter."""
    pad_size = kernel_size // 2
    padded_image = replicate_padding(image, pad_size)
    filtered_image = np.zeros_like(image)
    kernel = gaussian_kernel(kernel_size, sigma)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(neighborhood * kernel)

    return filtered_image.astype(np.uint8)

def median_filter(image, kernel_size=5):
    """Apply a median filter."""
    pad_size = kernel_size // 2
    padded_image = replicate_padding(image, pad_size)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.median(neighborhood)

    return filtered_image.astype(np.uint8)

# Load a sample grayscale image
image_path = "../images/catty.jpg"


image_RGB = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
gray_image = gray.rgb_to_grayscale(image_RGB)
image = NoiseAdder.add_gaussian_noise(gray_image)
# Apply filters
avg_filtered = average_filter(image, kernel_size=3)
gaussian_filtered = gaussian_filter(image, kernel_size=3, sigma=1)
median_filtered = median_filter(image, kernel_size=3)

# Display results
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[1].imshow(avg_filtered, cmap='gray')
axes[1].set_title("Average Filtered")
axes[2].imshow(gaussian_filtered, cmap='gray')
axes[2].set_title("Gaussian Filtered")
axes[3].imshow(median_filtered, cmap='gray')
axes[3].set_title("Median Filtered")

for ax in axes:
    ax.axis("off")

plt.show()
