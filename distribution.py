import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "CV/Image-Editor-Computer-Vision/images/gray.jpg"
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

# Load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Replace with your image path

# Compute histogram
hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

# Compute normalized distribution (PDF)
pdf = hist / hist.sum()

# Plot both histogram and distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Count")
plt.plot(hist, color="black")

plt.subplot(1, 2, 2)
plt.title("Distribution (PDF)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Probability")
plt.plot(pdf, color="blue")

plt.show()
