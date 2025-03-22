import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to grayscale
image = cv2.imread("../images/bubbles.jpg")  # Change this to your test image path
if image is None:
    raise FileNotFoundError("Image not found. Check the file path!")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges using Canny
edges = cv2.Canny(gray, 150, 200)

# Apply OpenCV Hough Line Transform
lines = cv2.HoughLines(edges, 15, theta=np.deg2rad(15) , threshold=500)  # 100 is the threshold
image_lines = image.copy()

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(image_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Apply OpenCV Hough Circle Transform
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=100
)

image_circles = image.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(image_circles, (x, y), r, (255, 0, 0), 2)

# Display results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Canny Edges")
plt.imshow(edges, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Hough Lines (OpenCV)")
plt.imshow(cv2.cvtColor(image_lines, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Hough Circles (OpenCV)")
plt.imshow(cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB))

plt.show()
