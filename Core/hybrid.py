import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    k = (size - 1) // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def convolve2d(image, kernel):
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    pad_height = k_height // 2
    pad_width = k_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image)

    for i in range(i_height):
        for j in range(i_width):
            region = padded_image[i:i + k_height, j:j + k_width]
            output[i, j] = np.sum(region * kernel)

    return output

def resize_images(image1, image2):
    """ Resize the larger image to match the smaller image's dimensions. """
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    target_height = min(h1, h2)
    target_width = min(w1, w2)

    image1_resized = cv2.resize(image1, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    image2_resized = cv2.resize(image2, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    return image1_resized, image2_resized

def hybrid_image(image1, image2, sigma=8):
    """ Create a hybrid image by combining low-pass filtered image1 with high-pass filtered image2. """
    image1, image2 = resize_images(image1, image2)

    kernel_size = 30
    gaussian_k = gaussian_kernel(kernel_size, sigma)

    low_frequencies = convolve2d(image1, gaussian_k)
    blurred_image2 = convolve2d(image2, gaussian_k)
    high_frequencies = image2 - blurred_image2

    hybrid = low_frequencies + high_frequencies
    hybrid = np.clip(hybrid, 0, 255)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(low_frequencies, cmap='gray')
    axes[0].set_title('Low Frequencies (Image 1)')
    axes[0].axis('off')

    axes[1].imshow(high_frequencies + 127.5, cmap='gray')
    axes[1].set_title('High Frequencies (Image 2)')
    axes[1].axis('off')

    axes[2].imshow(hybrid, cmap='gray')
    axes[2].set_title('Hybrid Image')
    axes[2].axis('off')

    # plt.show()

    return hybrid.astype(np.uint8)

# # Example usage:
# image1 = cv2.imread('path_to_image1.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('path_to_image2.jpg', cv2.IMREAD_GRAYSCALE)
# hybrid = hybrid_image(image1, image2)
