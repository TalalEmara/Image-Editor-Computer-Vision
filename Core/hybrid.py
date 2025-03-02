import numpy as np

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize

def laplacian_kernel():
    """Generate a 3x3 Laplacian kernel."""
    return np.array([[0,  1,  0],
                     [1, -4,  1],
                     [0,  1,  0]])

def convolve(image, kernel):
    """Manually apply a convolution operation."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2  # Padding size

    # Pad the image with reflection to avoid boundary artifacts
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Output image
    result = np.zeros_like(image, dtype=np.float32)

    # Apply convolution
    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i + k_h, j:j + k_w]  # Extract region
            result[i, j] = np.sum(region * kernel)  # Element-wise multiply and sum

    return result

def hybrid_image(image1, image2, sigma=10, alpha=1.5):
    """Create a hybrid image by combining a low-pass and high-pass filtered image."""
    # Adjust kernel size dynamically
    kernel_size = int(6 * sigma) | 1  # Ensure it's odd
    gauss_kernel = gaussian_kernel(kernel_size, sigma)

    # Apply filters
    low_frequencies = convolve(image1, gauss_kernel)
    high_frequencies = image2 - convolve(image2, gauss_kernel)

    # Normalize and scale high frequencies
    high_frequencies = alpha * (high_frequencies - high_frequencies.min()) / (high_frequencies.max() - high_frequencies.min()) * 255

    # Combine both images
    hybrid = low_frequencies + high_frequencies

    return np.clip(hybrid, 0, 255).astype(np.uint8)
