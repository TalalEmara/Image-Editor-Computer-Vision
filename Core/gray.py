import cv2
import numpy as np
import matplotlib.pyplot as plt

# image_path = "images/colored.jpg"


# image_RGB = cv2.imread(image_path, cv2.COLOR_BGR2RGB)



def rgb_to_grayscale(image):
    """Convert an RGB image to grayscale using luminosity method."""
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b  # Standard grayscale conversion
    return grayscale.astype(np.uint8)


def normalize_image(image):
    """Apply zero mean and unit variance normalization."""
    mean_val = np.mean(image)  # Compute mean intensity
    std_val = np.std(image)  # Compute standard deviation

    if std_val == 0:  # Avoid division by zero
        std_val = 1

    normalized_image = (image - mean_val) / std_val  # Zero mean normalization

    normalized_image = ((normalized_image - normalized_image.min()) /
                        (normalized_image.max() - normalized_image.min())) * 255

    return normalized_image.astype(np.uint8)

def prepare_for_display(image):
    """Scale a zero-mean image into [0,255] while keeping contrast."""
    min_val, max_val = np.min(image), np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)


# # Convert to grayscale
# grayscale_img = rgb_to_grayscale(image_RGB)


# # Apply normalization
# normalized_img = normalize_image(grayscale_img)
# display_img = prepare_for_display(normalized_img)

# # Display results
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(grayscale_img, cmap='gray')
# axes[0].set_title("Grayscale Image")
# axes[1].imshow(normalized_img, cmap='gray')
# axes[1].set_title("Normalized Image")

# axes[2].imshow(normalized_img, cmap='seismic')

# axes[2].set_title("Normalized Image (Without Rescaling)")

# for ax in axes:
#     ax.axis("off")

# plt.show()
