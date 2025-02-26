import cv2
import numpy as np


def globalThreshold(image ,threshold ):
    # threshold is a parameter from 0 to 255
    biImage = np.zeros_like(image)
    biImage[image > threshold] = 255

    return biImage



def sauvolaThresholding(image, window_size=100, k=0.3, R=128):
    """
    Perform Sauvola thresholding efficiently using integral images.

    Parameters:
    - image: Grayscale image as a NumPy array.
    - window_size: Local window size (must be an odd integer).
    - k: Parameter controlling threshold level.
    - R: Dynamic range of standard deviation (default is 128 for 8-bit images).

    Returns:
    - Binary image.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer. Try again, wise one!")

    image = image.astype(np.float64)

    # Padding to avoid border issues
    pad_size = window_size // 2
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)

    # Compute integral images for fast summation
    integral_img = cv2.integral(padded_image)
    integral_sq_img = cv2.integral(padded_image ** 2)

    # Compute sum and squared sum using integral images
    y, x = np.indices(image.shape)  # Get all coordinates

    x1, y1 = x, y
    x2, y2 = x + window_size, y + window_size

    sum_ = (integral_img[y2, x2] - integral_img[y1, x2] -
            integral_img[y2, x1] + integral_img[y1, x1])

    sum_sq = (integral_sq_img[y2, x2] - integral_sq_img[y1, x2] -
              integral_sq_img[y2, x1] + integral_sq_img[y1, x1])

    N = window_size * window_size
    mean = sum_ / N

    # Ensure variance is non-negative to prevent sqrt nightmares
    variance = (sum_sq - (sum_**2) / N) / N
    variance = np.maximum(variance, 0)
    stddev = np.sqrt(variance)

    # Compute Sauvola threshold
    threshold = mean * (1 + k * ((stddev / R) - 1))

    # Apply thresholding
    binary_image = (image > threshold).astype(np.uint8) * 255

    return binary_image
