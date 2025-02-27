import cv2
import numpy as np
import matplotlib.pyplot as plt
from Core.gray import rgb_to_grayscale

image_path = "CV/Image-Editor-Computer-Vision/images/catty.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
imageRGB = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

def histogramGS(image):
    """
    image: an RGB image (3D numpy array) or a grayscale image (2D numpy array)
    returns: grayScale (x-axis), histogram (y-axis)
    """
    if image.ndim == 3:
        imageGS = rgb_to_grayscale(image)
        grayScale, histogram = np.unique(imageGS, return_counts=True)
    else:
        grayScale, histogram = np.unique(image, return_counts=True)
    return grayScale, histogram

def distribution(image):
    """
    image: an RGB image (3D numpy array)
    returns: grayScale (x-axis), distribution (y-axis)
    """
    imageGS = rgb_to_grayscale(image)
    grayScale, histogram = np.unique(imageGS, return_counts=True)
    distribution = histogram / np.sum(histogram)
    return grayScale, distribution

def histogramRGB(image):
    """
    image: an RGB image (3D numpy array)
    returns: red, green, blue histograms (1D numpy arrays)
    """
    red = image[:, :, 0].flatten()
    green = image[:, :, 1].flatten()
    blue = image[:, :, 2].flatten()
    _, redHist = histogramGS(red)
    _, greenHist = histogramGS(green)
    _, blueHist = histogramGS(blue)
    return redHist, greenHist, blueHist


def show_histograms(image):
    fig, axes = plt.subplots(3, 1)

    # Plot 1: Grayscale Histogram
    gs, hg = histogramGS(image)
    axes[0].bar(gs, hg, color='gray')
    axes[0].set_title("Histogram of Image")
    axes[0].set_xlabel("Gray Scale")
    axes[0].set_ylabel("Frequency")

    # Plot 2: Grayscale Distribution (PDF)
    gs, dist = distribution(image)
    axes[1].bar(gs, dist, color='lightgreen')
    axes[1].set_title("Distribution of Image")
    axes[1].set_xlabel("Gray Scale")
    axes[1].set_ylabel("PDF")

    # Plot 3: RGB Histogram
    red, green, blue = histogramRGB(imageRGB)
    axes[2].plot(red, color="red", label="Red")
    axes[2].plot(green, color="green", label="Green")
    axes[2].plot(blue, color="blue", label="Blue")
    axes[2].set_title("RGB Histogram")
    axes[2].set_xlabel("RGB")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# show_histograms(imageRGB)