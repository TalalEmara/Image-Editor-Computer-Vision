import cv2
import numpy as np


def add_uniform_noise(image, noise_range=(-50, 50), grayscale=False):
    """
    Adds uniform noise to an image.

    :param image: Input image (NumPy array).
    :param noise_range: Tuple (a, b) specifying the range of noise values.
    :param grayscale: If True, applies noise to a grayscale image.
    :return: Noisy image.
    """
    noisy_image = image.copy()
    a, b = noise_range

    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noise = np.random.uniform(a, b, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy_image


def add_gaussian_noise(image, mean=0, std_dev=50):
    """
    Adds Gaussian noise to an image.

    :param image: Input image (NumPy array).
    :param mean: Mean (μ) of Gaussian distribution.
    :param std_dev: Standard deviation (σ) controlling noise intensity.
    :return: Noisy image.
    """
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return noisy_image


def add_salt_pepper_noise(image, prob=0.01, salt_ratio=0.5):
    """
    Adds salt & pepper noise to an image.

    :param image: Input image (NumPy array).
    :param prob: Probability of noise (e.g., 0.01 for 1% noise).
    :param salt_ratio: Ratio of salt (white) vs. pepper (black) noise.
    :return: Noisy image.
    """
    noisy_image = image.copy()
    total_pixels = image.size

    num_salt = int(total_pixels * prob * salt_ratio)
    num_pepper = int(total_pixels * prob * (1 - salt_ratio))

    # Add salt (white) noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Add pepper (black) noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image
