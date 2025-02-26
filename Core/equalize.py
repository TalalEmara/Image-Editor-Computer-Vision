import numpy as np
import matplotlib.pyplot as plt
from Core.histogram import histogramGS, distribution, histogramRGB


def equalization(image):
    grayScale, histogram = histogramGS(image)

    # Calculate CDF
    cdf = np.cumsum(histogram)
    
    # Mask all pixels in the CDF with '0' intensity
    cdf_masked = np.ma.masked_equal(cdf, 0)
    
    # Equalize the histogram by scaling the CDF
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    
    # Fill masked pixels with '0'
    cdf = np.ma.filled(cdf_masked, 0).astype('uint8')
    
    equalized_image = cdf[image]
    
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
    plt.show()
