import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.draw import disk

def draw_line(image, x1, y1, x2, y2, color=(0, 255, 0)):
    h, w, _ = image.shape  # Get image dimensions
    if not (0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h):
        return  # Ensure points are within bounds

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < w and 0 <= y1 < h:
            image[y1, x1] = color  # Assign (B, G, R) safely
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def draw_circle(image, xc, yc, r, color=255):
    x = 0
    y = r
    d = 3 - 2 * r

    def plot_circle_points(xc, yc, x, y):
        points = [(xc + x, yc + y), (xc - x, yc + y), (xc + x, yc - y), (xc - x, yc - y),
                  (xc + y, yc + x), (xc - y, yc + x), (xc + y, yc - x), (xc - y, yc - x)]
        for px, py in points:
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                image[py, px] = color

    while y >= x:
        plot_circle_points(xc, yc, x, y)
        x += 1
        if d > 0:
            y -= 1
            d += 4 * (x - y) + 10
        else:
            d += 4 * x + 6

def hough_lines(edges, theta_res=15, rho_res=15):
    height, width = edges.shape
    max_rho = int(np.hypot(height, width))
    rhos = np.arange(-max_rho, max_rho + rho_res, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    edge_points = np.argwhere(edges)
    for y, x in edge_points:
        for t_idx, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int((rho + max_rho) / rho_res)  # Fix indexing
            if 0 <= rho_idx < len(rhos):
                accumulator[rho_idx, t_idx] += 1

    return accumulator, thetas, rhos



# def extract_peak_lines(accumulator, thetas, rhos, threshold):
#     peaks = np.argwhere(accumulator > threshold)
#     lines = [(rhos[r], thetas[t]) for r, t in peaks]
#     return lines
def extract_peak_lines(accumulator, thetas, rhos, threshold):
    peaks = np.argwhere(accumulator > threshold)
    sorted_indices = peaks[np.argsort(accumulator[peaks[:, 0], peaks[:, 1]])[::-1]]

    # Non-maximum suppression
    selected_lines = []
    for r, t in sorted_indices:
        if len(selected_lines) > 0:
            prev_rho, prev_theta = selected_lines[-1]
            if abs(prev_rho - rhos[r]) < 10 and abs(prev_theta - thetas[t]) < np.deg2rad(10):
                continue
        selected_lines.append((rhos[r], thetas[t]))

    return selected_lines


# def draw_detected_lines(image, lines):
#     img = image.copy()
#
#     for rho, theta in lines:
#         a, b = np.cos(theta), np.sin(theta)
#         x0, y0 = int(a * rho), int(b * rho)
#
#         # Calculate line endpoints using standard approach
#         # This is similar to how OpenCV draws them
#         x1 = int(x0 - 1000 * b)  # Note: using b, not -b
#         y1 = int(y0 + 1000 * a)
#         x2 = int(x0 + 1000 * b)  # Note: using b, not -b
#         y2 = int(y0 - 1000 * a)
#
#         # Draw line
#         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
#     return img


def draw_detected_lines(image, lines, max_lines=100):
    img = image.copy()
    h, w = img.shape[:2]

    # # Sort lines by accumulator value (if available) and take only the top max_lines
    # if len(lines) > max_lines:
    #     lines = lines[:max_lines]

    for rho, theta in lines:
        # Calculate a point on the line
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = int(a * rho), int(b * rho)

        # These calculations extend the line by a reasonable amount
        # instead of using ±1000 which goes far outside the image
        extension = max(h, w) // 2
        x1 = int(x0 + extension * (-b))
        y1 = int(y0 + extension * (a))
        x2 = int(x0 - extension * (-b))
        y2 = int(y0 - extension * (a))

        # Clip line to image boundaries
        # This uses the Cohen-Sutherland line clipping algorithm
        def clip_line(x1, y1, x2, y2, w, h):
            # Define region codes
            INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

            # Calculate region codes for both endpoints
            def compute_code(x, y):
                code = INSIDE
                if x < 0:
                    code |= LEFT
                elif x >= w:
                    code |= RIGHT
                if y < 0:
                    code |= TOP
                elif y >= h:
                    code |= BOTTOM
                return code

            code1 = compute_code(x1, y1)
            code2 = compute_code(x2, y2)

            while True:
                # Both endpoints inside the clip window
                if code1 == 0 and code2 == 0:
                    return True, x1, y1, x2, y2

                # Both endpoints outside the clip window on the same side
                if code1 & code2:
                    return False, 0, 0, 0, 0

                # Select an outside point
                code_out = code1 if code1 else code2

                # Find intersection point
                if code_out & TOP:
                    x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                    y = 0
                elif code_out & BOTTOM:
                    x = x1 + (x2 - x1) * (h - 1 - y1) / (y2 - y1)
                    y = h - 1
                elif code_out & RIGHT:
                    y = y1 + (y2 - y1) * (w - 1 - x1) / (x2 - x1)
                    x = w - 1
                elif code_out & LEFT:
                    y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
                    x = 0

                # Replace the outside point with the intersection point
                if code_out == code1:
                    x1, y1 = int(x), int(y)
                    code1 = compute_code(x1, y1)
                else:
                    x2, y2 = int(x), int(y)
                    code2 = compute_code(x2, y2)

        valid, nx1, ny1, nx2, ny2 = clip_line(x1, y1, x2, y2, w, h)
        if valid:
            draw_line(img, nx1, ny1, nx2, ny2, color=(0, 255, 0))
    return img

def hough_circles(edges, radius_range):
    height, width = edges.shape
    min_radius, max_radius = radius_range
    radii = np.arange(min_radius, max_radius + 1)
    accumulator = np.zeros((height, width, len(radii)), dtype=np.int32)
    edge_points = np.argwhere(edges)

    for y, x in edge_points:
        for r_idx, r in enumerate(radii):
            for theta in np.arange(0, 2 * np.pi, np.pi / 180):
                a, b = int(x - r * np.cos(theta)), int(y - r * np.sin(theta))
                if 0 <= a < width and 0 <= b < height:
                    accumulator[b, a, r_idx] += 1

    return accumulator, radii


def extract_peak_circles(accumulator, radii, threshold):
    peaks = np.argwhere(accumulator > threshold)
    circles = [(x, y, radii[r]) for y, x, r in peaks]
    return circles


def draw_detected_circles(image, circles):
    img = image.copy()
    for x, y, r in circles:
        # cv2.circle(img, (x, y), r, (255, 0, 0), 2)
        draw_circle(img, x, y, r, color=(255, 0, 0))

    return img


# Load image and detect edges
image = cv2.imread("../images/planets.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 150,200)

# # Apply Hough Line Transform
# acc_lines, thetas, rhos = hough_lines(edges)
# lines = extract_peak_lines(acc_lines, thetas, rhos, threshold=500)
# image_lines = draw_detected_lines(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), lines)

# Apply Hough Circle Transform
acc_circles, radii = hough_circles(edges, radius_range=(10, 100))
circles = extract_peak_circles(acc_circles, radii, threshold=150)
image_circles = draw_detected_circles(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), circles)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Edges")
plt.imshow(edges, cmap='gray')
# plt.subplot(1, 3, 2)
# plt.title("Detected Lines")
# plt.imshow(image_lines)
plt.subplot(1, 3, 3)
plt.title("Detected Circles")
plt.imshow(image_circles)
plt.show()
