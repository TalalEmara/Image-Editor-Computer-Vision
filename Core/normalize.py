import numpy as np
import matplotlib.pyplot as plt

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
