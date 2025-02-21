import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# image = cv2.imread(image_path)

# # Convert from BGR to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Split channels
# r, g, b = cv2.split(image_rgb)

# # Create histograms
# plt.figure(figsize=(10, 5))
# plt.title("Histogram of Image")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")

# # Plot histograms for each channel
# plt.hist(r.ravel(), bins=256, color='red', alpha=0.6, label='Red')
# plt.hist(g.ravel(), bins=256, color='green', alpha=0.6, label='Green')
# plt.hist(b.ravel(), bins=256, color='blue', alpha=0.6, label='Blue')

# plt.legend()
# plt.show()


# ## Gray Image Histogram
# imagegray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Replace with your image path

# # Compute histogram using NumPy
# hist, bins = np.histogram(imagegray.ravel(), bins=256, range=[0, 256])

# # Plot the histogram
# plt.figure(figsize=(10, 5))
# plt.title("Grayscale Image Histogram")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.plot(hist, color="black")  # Grayscale histogram
# plt.xlim([0, 256])
# plt.show()

# # Load the image in grayscale
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Replace with your image path

# # Compute histogram
# hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

# # Compute normalized distribution (PDF)
# pdf = hist / hist.sum()

# # Plot both histogram and distribution
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Histogram")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Count")
# plt.plot(hist, color="black")

# plt.subplot(1, 2, 2)
# plt.title("Distribution (PDF)")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Probability")
# plt.plot(pdf, color="blue")

# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load image in grayscale
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Compute histogram
# hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# # Normalize histogram for smooth curve
# hist_norm = hist / hist.sum()

# # Plot histogram
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.bar(bins[:-1], hist, width=1, color='gray', alpha=0.7)
# plt.title("Histogram")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")

# # Plot distribution curve (smoothed histogram)
# plt.subplot(1, 2, 2)
# sns.kdeplot(image.flatten(), color="black", bw_adjust=0.5)  # KDE smoothing
# plt.title("Distribution Curve")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Density")

# plt.tight_layout()
# plt.show()

image_path = "CV/Image-Editor-Computer-Vision/images/colored.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# print(image)

def histogram(image):
    """
    image: a grayscale image (2D numpy array)
    returns: grayScale (x-axis), histogram (y-axis)
    """
    grayScale, histogram = np.unique(image, return_counts=True)
    return grayScale, histogram


# Testing
gs, hg = histogram(image)
plt.bar(gs, hg)
plt.title("Histogram of Image")
plt.xlabel("Gray Scale")
plt.ylabel("Frequency")
plt.show()
