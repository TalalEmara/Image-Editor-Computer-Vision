
import numpy as np
import matplotlib.pyplot as plt
from Core.histogram import histogramGS, distribution, histogramRGB
from Core.gray import rgb_to_grayscale  

def cumulative_summation(histogram):
    return np.cumsum(histogram)

def rgb_to_yuv(image):
    """Convert an RGB image to YUV color space using NumPy."""
    matrix = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615, -0.51499, -0.10001]])
    return np.dot(image, matrix.T)

def yuv_to_rgb(yuv):
    """Convert a YUV image back to RGB using NumPy."""
    matrix = np.array([[1.0, 0.0, 1.13983],
                       [1.0, -0.39465, -0.58060],
                       [1.0, 2.03211, 0.0]])
    return np.dot(yuv, matrix.T)

def equalization(image):
    """
    Applies histogram equalization to grayscale and color images (using YUV space for color).
    
    Parameters:
    - image: numpy array (grayscale or RGB)
    
    Returns:
    - equalized_image: numpy array (same shape as input)
    """
    
    if image.ndim == 2:  # Grayscale image
        imageGS = image  # Already grayscale
    else:  # Convert RGB to grayscale for histogram computation
        imageGS = rgb_to_grayscale(image)

    histogram, _ = np.histogram(imageGS.flatten(), 256, [0, 256])

    # Compute CDF
    cdf = cumulative_summation(histogram)
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = np.ma.filled(cdf, 0).astype(np.uint8)

    if image.ndim == 2:  # Grayscale image equalization
        equalized_image = cdf[imageGS]
        return equalized_image
    else:  # Color image, process in YUV space
        yuv = rgb_to_yuv(image.astype(np.float32) / 255.0)  # Normalize to [0, 1]

        Y_channel = (yuv[:, :, 0] * 255).astype(np.uint8)  # Extract Y channel (brightness)

        # Equalize the Y channel
        Y_equalized = cdf[Y_channel] / 255.0  # Normalize back to [0, 1]

        # Replace equalized Y channel back
        yuv[:, :, 0] = Y_equalized

        # Convert back to RGB
        equalized_image = yuv_to_rgb(yuv) * 255  # Scale back to 0-255
        equalized_image = np.clip(equalized_image, 0, 255).astype(np.uint8)  # Ensure valid range

        return equalized_image


def show_equalized_histograms(equalized_image):
    
    fig, axes = plt.subplots(4, 1)

    # Equalized Grayscale Histogram
    gs_eq, hist_eq = histogramGS(equalized_image)
    axes[0].bar(gs_eq, hist_eq, color='gray', width=1)
    axes[0].set_title("Histogram of Equalized Image")
    axes[0].set_xlabel("Gray Scale")
    axes[0].set_ylabel("Frequency")

    # Equalized Grayscale Distribution 
    gs_eq, dist_eq = distribution(equalized_image)
    axes[1].bar(gs_eq, dist_eq, color='lightgreen', width=1)
    axes[1].set_title("Distribution of Equalized Image")
    axes[1].set_xlabel("Gray Scale")
    axes[1].set_ylabel("PDF")

    #CDF graph
    cdf = np.cumsum(hist_eq)  
    cdf_normalized = cdf / cdf.max()
    axes[2].plot(gs_eq, cdf, color='blue', linewidth=2)
    axes[2].set_title("Cumulative Distribution Function (CDF)")
    axes[2].set_xlabel("Gray Scale")
    axes[2].set_ylabel("Cumulative Probability")

    # Equalized RGB Histogram 
    red_eq, green_eq, blue_eq = histogramRGB(equalized_image)
    axes[3].plot(red_eq, color="red", label="Red")
    axes[3].plot(green_eq, color="green", label="Green")
    axes[3].plot(blue_eq, color="blue", label="Blue")
    axes[3].set_title("RGB Histogram")
    axes[3].set_xlabel("RGB")
    axes[3].set_ylabel("Frequency")
    axes[3].legend()

    plt.tight_layout()
    plt.ion()
    plt.show()
