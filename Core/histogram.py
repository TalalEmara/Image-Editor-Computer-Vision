import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
import pyqtgraph as pg


from Core.gray import rgb_to_grayscale


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


from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

def show_histograms(image):
    widget = QWidget()
    layout = QVBoxLayout(widget)

    # Create a Matplotlib figure
    fig, axes = plt.subplots(3, 1, figsize=(5, 8))
    fig.patch.set_alpha(0)

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
    red, green, blue = histogramRGB(image)
    axes[2].plot(red, color="red", label="Red")
    axes[2].plot(green, color="green", label="Green")
    axes[2].plot(blue, color="blue", label="Blue")
    axes[2].set_title("RGB Histogram")
    axes[2].set_xlabel("RGB")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()

    # Convert Matplotlib figure to a PyQt widget
    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)

    return widget

def get_histogram_widget(image):
    """Creates a PyQtGraph widget showing grayscale histogram, grayscale distribution, and RGB histogram."""
    widget = QWidget()
    layout = QVBoxLayout(widget)

    # Create PyQtGraph plot widgets
    gray_hist_plot = pg.PlotWidget(title="Histogram of Image")
    gray_dist_plot = pg.PlotWidget(title="Distribution of Image")
    rgb_hist_plot = pg.PlotWidget(title="RGB Histogram")

    # Add plots to layout
    layout.addWidget(gray_hist_plot)
    layout.addWidget(gray_dist_plot)
    layout.addWidget(rgb_hist_plot)

    # Grayscale Histogram using BarGraphItem
    gs, hg = histogramGS(image)
    bars = pg.BarGraphItem(x=gs, height=hg, width=1, brush="gray",  pen=None)
    gray_hist_plot.addItem(bars)
    gray_hist_plot.setLabel('bottom', 'Gray Scale')
    gray_hist_plot.setLabel('left', 'Frequency')

    # Grayscale Distribution (PDF)
    gs, dist = distribution(image)
    bars_dist = pg.BarGraphItem(x=gs, height=dist, width=1, brush='lightgreen',  pen=None)
    gray_dist_plot.addItem(bars_dist)
    gray_dist_plot.setLabel('bottom', 'Gray Scale')
    gray_dist_plot.setLabel('left', 'PDF')

    # RGB Histogram using BarGraphItem
    red, green, blue = histogramRGB(image)
    x = np.arange(len(red))

    red_bars = pg.BarGraphItem(x=x, height=red, width=1, brush=pg.mkBrush(255, 0, 0, 150), pen=None)
    green_bars = pg.BarGraphItem(x=x, height=green, width=1, brush=pg.mkBrush(0, 255, 0, 150), pen=None)
    blue_bars = pg.BarGraphItem(x=x, height=blue, width=1, brush=pg.mkBrush(0, 0, 255, 150), pen=None)



    rgb_hist_plot.addItem(red_bars)
    rgb_hist_plot.addItem(green_bars)
    rgb_hist_plot.addItem(blue_bars)

    rgb_hist_plot.setLabel('bottom', 'RGB')
    rgb_hist_plot.setLabel('left', 'Frequency')

    return widget

# show_histograms(imageRGB)