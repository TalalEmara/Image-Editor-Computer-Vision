import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_gradients(edges):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = cv2.filter2D(edges, -1, sobel_x)
    grad_y = cv2.filter2D(edges, -1, sobel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)
    return magnitude, direction


def hough_circles(edges, radius_range):
    height, width = edges.shape
    min_radius, max_radius = radius_range
    radii = np.arange(min_radius, max_radius + 1)
    accumulator = np.zeros((height, width, len(radii)), dtype=np.float32)

    magnitude, direction = compute_gradients(edges)
    edge_points = np.argwhere(edges > 0)

    for y, x in edge_points:
        theta = direction[y, x]
        for r_idx, r in enumerate(radii):
            a = int(x - r * np.cos(theta))
            b = int(y - r * np.sin(theta))
            if 0 <= a < width and 0 <= b < height:
                accumulator[b, a, r_idx] += magnitude[y, x]

    return accumulator, radii


def extract_peak_circles(accumulator, radii, threshold=0.5):
    circles = []
    max_accum = np.max(accumulator)
    threshold_value = threshold * max_accum
    peaks = np.argwhere(accumulator > threshold_value)

    for y, x, r_idx in peaks:
        circles.append((x, y, radii[r_idx]))

    return circles


def draw_detected_circles(image, circles):
    img = image.copy()
    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (255, 0, 0), 2)
    return img


# Load image and detect edges
image = cv2.imread("../images/bubbles.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 150, 200)

# Apply Hough Circle Transform
acc_circles, radii = hough_circles(edges, radius_range=(10, 100))
circles = extract_peak_circles(acc_circles, radii, threshold=0.8)
image_circles = draw_detected_circles(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), circles)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Edges")
plt.imshow(edges, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Detected Circles")
plt.imshow(image_circles)
plt.show()
